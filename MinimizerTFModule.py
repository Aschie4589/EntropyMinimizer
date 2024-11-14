import tensorflow as tf

class MinimizerTFModule(tf.Module):
    '''
    This class is designed to work with quantum information theory, specifically for minimizing
    entropy-like functions over quantum channels using Kraus operators.
    It supports parallelization and iterative algorithmic steps to update quantum state vectors.
    '''

    def __init__(self, kraus_list, epsilon, parallel_computations=1, vec_states=None):
        '''
        Initializes the MinimizerTFModule with Kraus operators, step size (epsilon), and parallel computations.

        Parameters:
        kraus_list (list of tf.Tensor): List of Kraus operators for the quantum channel.
        epsilon (float): Step size for the update rule in the algorithm.
        parallel_computations (int): The number of quantum states to process in parallel.
        vec_states (tf.Tensor or None): Initial quantum state vectors. If None, random vectors are initialized.
        '''
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

    @tf.function
    def initialize_random_vectors(self):
        '''
        Initializes random quantum state vectors by drawing from a standard normal distribution.
        The vectors are normalized to have unit norm.

        Returns:
        tf.Tensor: A batch of normalized random quantum state vectors.
        '''
        # Generate random real and imaginary components for the quantum states
        random_real = tf.random.normal(shape=[self.parallel_computations, self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        random_imaginary = tf.random.normal(shape=[self.parallel_computations, self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        
        # Combine real and imaginary parts to create complex numbers
        random_complex = tf.complex(random_real, random_imaginary)
        
        # Normalize the complex vectors (each vector should have unit norm)
        return tf.linalg.normalize(random_complex, axis=1)[0]  # Normalize returns both normalized vectors and their norms

    @tf.function
    def get_projectors(self, v):
        '''
        Computes the projectors for a batch of quantum state vectors.

        Parameters:
        v (tf.Tensor): A tensor of quantum state vectors, shape [m, n, 1] where m is the number of vectors and n is the vector dimension.

        Returns:
        tf.Tensor: A tensor of projectors, shape [1, m, n, n].
        '''
        return tf.expand_dims(tf.matmul(v, v, adjoint_b=True), axis=0)

    @tf.function
    def apply_channel(self, mat, kraus):
        '''
        Applies the quantum channel to a matrix using the Kraus operators.

        The operation is defined as:
        result = sum_k (K_k * mat * K_k†)

        Parameters:
        mat (tf.Tensor): The matrix (quantum state or density matrix) to which the channel is applied.
        kraus (tf.Tensor): The tensor of Kraus operators.

        Returns:
        tf.Tensor: The matrix after the channel is applied.
        '''
        return tf.reduce_sum(tf.matmul(tf.matmul(kraus, mat), kraus, adjoint_b=True), axis=0)

    @tf.function
    def apply_dual_channel(self, mat, kraus):
        '''
        Applies the dual (adjoint) of the quantum channel to a matrix.

        The operation is defined as:
        result = sum_k (K_k† * mat * K_k)

        Parameters:
        mat (tf.Tensor): The matrix (quantum state or density matrix) to which the dual channel is applied.
        kraus (tf.Tensor): The tensor of Kraus operators.

        Returns:
        tf.Tensor: The matrix after the dual channel is applied.
        '''
        return tf.reduce_sum(tf.matmul(tf.matmul(kraus, mat, adjoint_a=True), kraus), axis=0)

    @tf.function
    def algorithm_step(self, mat, kraus, epsilon):
        '''
        Performs a single step of the entropy-minimizing algorithm to update the quantum state vector.

        The update is based on applying the channel and its dual, taking a logarithm, and using the eigenvector corresponding
        to the largest eigenvalue as the updated quantum state.

        Parameters:
        mat (tf.Tensor): The matrix (quantum state or density matrix) to be updated.
        kraus (tf.Tensor): The tensor of Kraus operators.
        epsilon (tf.Tensor): A small constant for the update rule.

        Returns:
        tf.Tensor: The updated quantum state vector.
        '''
        # Compute the matrix logarithm of the updated state
        new_mat = (1 - epsilon) * self.apply_dual_channel(
            tf.linalg.logm(
                (1 - epsilon) * self.apply_channel(mat, kraus) + epsilon / tf.cast(self.input_dim, tf.complex128) * tf.eye(self.input_dim, dtype=tf.complex128, batch_shape=[self.parallel_computations])
            ), kraus
        ) + epsilon / tf.cast(self.output_dim, tf.complex128) * tf.eye(self.output_dim, dtype=tf.complex128, batch_shape=[self.parallel_computations])

        # Return the eigenvector corresponding to the largest eigenvalue
        return tf.expand_dims(tf.linalg.eigh(new_mat)[1][..., -1], [-1])

    @tf.function
    def current_entropy(self, mat, kraus, epsilon):
        '''
        Computes the entropy of the quantum state after applying the channel.

        Entropy is computed as the von Neumann entropy of the density matrix:
        S = - sum (λ_i * log(λ_i))

        Parameters:
        mat (tf.Tensor): The matrix (quantum state or density matrix).
        kraus (tf.Tensor): The tensor of Kraus operators.
        epsilon (tf.Tensor): A small constant to adjust the matrix.

        Returns:
        tf.Tensor: The entropy of the quantum state.
        '''
        applied = (1 - epsilon) * self.apply_channel(mat, kraus) + epsilon / tf.cast(self.input_dim, tf.complex128) * tf.eye(self.input_dim, dtype=tf.complex128, batch_shape=[self.parallel_computations])
        eig = tf.linalg.eigvalsh(applied)
        log_eig = tf.math.log(eig)
        return -tf.reduce_sum(eig * log_eig, axis=[-1])

    @tf.function
    def step(self):
        '''
        Performs a single step of the iterative algorithm by updating the quantum state vector.

        The quantum state vector is updated by applying the algorithm step, which minimizes entropy.

        Parameters:
        None

        Returns:
        None. The state vector is updated in place.
        '''
        self.vector.assign(self.algorithm_step(self.get_projectors(self.vector), self.kraus_ops, self.epsilon))
