import tensorflow as tf

class MinimizerModule(tf.Module):

    def __init__(self, kraus_list, epsilon, parallel_computations=1, vec_states=None):
        # Set epsilon and parallel computations as constants
        self.epsilon = tf.constant(epsilon, dtype=tf.complex128)
        self.parallel_computations = tf.constant(parallel_computations, dtype=tf.int32)

        # Stack Kraus operators and expand dimensions for further computation
        self.kraus_ops = tf.constant(tf.expand_dims(tf.stack(kraus_list), axis=1))  # Shape: (N, 1, d_out, d_in)
        
        # Input and output dimensions based on the Kraus operators
        self.input_dim = tf.constant(self.kraus_ops.shape[3], dtype=tf.int32)
        self.output_dim = tf.constant(self.kraus_ops.shape[2], dtype=tf.int32)

        # Initialize state vector, either from provided vec_states or by generating random vectors
        if isinstance(vec_states, type(None)):
            self.vector = tf.Variable(self.initialize_random_vectors())
        else:
            self.vector = tf.Variable(initial_value=vec_states)

        # Initialize constants needed throughout
        self.binary_entropy = tf.constant(-tf.abs(self.epsilon)*tf.math.log(tf.abs(self.epsilon))-(1-tf.abs(self.epsilon))*tf.math.log(1-tf.abs(self.epsilon)), dtype=tf.float64)
        self.entropy_error = self.binary_entropy/(2*(1-tf.abs(self.epsilon)))
        # Initialize the entropy vectors
        self.entropy = tf.Variable(initial_value=self.current_entropy(self.get_projectors(self.vector),self.kraus_ops,self.epsilon))
        self.estimated_entropy = tf.Variable(initial_value=(self.entropy - tf.abs(self.epsilon)*tf.math.log(tf.cast(self.output_dim, dtype=tf.float64)))/(tf.abs(1-self.epsilon)) - self.entropy_error)
        self.ub_entropy = tf.Variable(initial_value=self.estimated_entropy + self.entropy_error)
        self.lb_entropy = tf.Variable(initial_value=self.estimated_entropy - self.entropy_error)

    @tf.function
    def initialize_random_vectors(self):
        # Generate random real and imaginary components for the quantum states
        random_real = tf.random.normal(shape=[self.parallel_computations, self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        random_imaginary = tf.random.normal(shape=[self.parallel_computations, self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        
        # Combine real and imaginary parts to create complex numbers
        random_complex = tf.complex(random_real, random_imaginary)
        
        # Normalize the complex vectors (each vector should have unit norm)
        return tf.linalg.normalize(random_complex, axis=1)[0]  # Normalize returns both normalized vectors and their norms

    @tf.function
    def get_projectors(self, v):
        return tf.expand_dims(tf.matmul(v, v, adjoint_b=True), axis=0)

    @tf.function
    def apply_channel(self, mat, kraus):
        return tf.reduce_sum(tf.matmul(kraus, tf.matmul(mat, kraus, adjoint_b=True)), axis=0)

    @tf.function
    def apply_dual_channel(self, mat, kraus):
        return tf.reduce_sum(tf.matmul(kraus, tf.matmul(mat,kraus), adjoint_a=True), axis=0)

    @tf.function
    def algorithm_step(self, mat, kraus, epsilon):
        new_mat = (1 - epsilon) * self.apply_dual_channel(tf.linalg.logm((1 - epsilon) * self.apply_channel(mat, kraus) + epsilon / tf.cast(self.input_dim, tf.complex128) * tf.eye(self.input_dim, dtype=tf.complex128, batch_shape=[self.parallel_computations])), kraus) + epsilon / tf.cast(self.output_dim, tf.complex128) * tf.eye(self.output_dim, dtype=tf.complex128, batch_shape=[self.parallel_computations])
        # Return the eigenvector corresponding to the largest eigenvalue
        return tf.expand_dims(tf.linalg.eigh(new_mat)[1][..., -1], [-1])

    @tf.function
    def current_entropy(self, mat, kraus, epsilon):
        eig =  tf.abs(tf.linalg.eigvalsh((1 - epsilon) * self.apply_channel(mat, kraus) + epsilon / tf.cast(self.input_dim, tf.complex128) * tf.eye(self.input_dim, dtype=tf.complex128, batch_shape=[self.parallel_computations])))
        log_eig = tf.math.log(eig)
        return -tf.reduce_sum(eig * log_eig, axis=[-1])

    @tf.function
    def update_entropies(self, mat, kraus):
        # Update the current entropy
        self.entropy.assign(self.current_entropy(mat,kraus, self.epsilon))
        # Update the estimated entropy
        self.estimated_entropy.assign((self.entropy - tf.abs(self.epsilon)*tf.math.log(tf.cast(self.output_dim, dtype=tf.float64)))/tf.abs(1-self.epsilon) - self.entropy_error)
        # Update the upper and lower bounds
        self.ub_entropy.assign(self.estimated_entropy + self.entropy_error)
        self.lb_entropy.assign(self.estimated_entropy - self.entropy_error)

    def step(self):

        self.vector.assign(self.algorithm_step(self.get_projectors(self.vector), self.kraus_ops, self.epsilon))
        # Update the numerical values of entropy and the related estimated quantities
        self.update_entropies(self.get_projectors(self.vector),self.kraus_ops)