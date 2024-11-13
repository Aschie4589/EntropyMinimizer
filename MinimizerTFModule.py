import tensorflow as tf

class MinimizerTFModule(tf.Module):
    '''
    I need to store the kraus operators of the channel, precompute kraus operators of the dual channel
    the input and output dimensions (so I don't have to compute them all the time)

    I need to implement stepping through the algorithm.
    '''

    def __init__(self, kraus_list, epsilon, parallel_computations=1, vec_states = None):
        self.epsilon=tf.constant(epsilon,dtype=tf.complex128)
        self.parallel_computations = tf.constant(parallel_computations,dtype=tf.int32)

        self.kraus_ops = tf.constant(tf.expand_dims(tf.stack(kraus_list), axis=1))  # Shape: (N, 1, d_out, d_in)
        self.input_dim = tf.constant(self.kraus_ops.shape[3],dtype=tf.int32)
        self.output_dim = tf.constant(self.kraus_ops.shape[2],dtype=tf.int32)

        if isinstance(vec_states,type(None)):
            self.vector = tf.Variable(self.initialize_random_vectors())
        else:   
            self.vector = tf.Variable(initial_value=vec_states)
        pass

    @tf.function
    def initialize_random_vectors(self):
        #Choosing standard gaussian for both real and imaginary parts gives the uniform distribution in the state space!
        random_real = tf.random.normal(shape=[self.parallel_computations,self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        random_imaginary = tf.random.normal(shape=[self.parallel_computations,self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        random_complex = tf.complex(random_real, random_imaginary)
        return tf.linalg.normalize(random_complex,axis=1)[0] #normalize returns both the normalized vectors and the previous norms

    def get_projectors(self,v):
        #v is a collection of m (n,1) vectors = a tensor of dim[m,n,1]
        #returns ar [1,m,n,n] tensor
        return tf.expand_dims(tf.matmul(v, v, adjoint_b=True),axis=0)

    def apply_channel(self,mat,kraus):
        return tf.reduce_sum(tf.matmul(tf.matmul(kraus, mat), kraus, adjoint_b=True), axis=0)  # Shape: [n, d, d]

    def apply_dual_channel(self,mat,kraus):
        #kraus are assomed to be of shape [r, 1, dout, din] where r is the number of kraus ops
        #mat is assumed to be of shape [1, n, din, din] where n is number of matrices to process at once

        return tf.reduce_sum(tf.matmul(tf.matmul(kraus, mat,adjoint_a=True), kraus), axis=0)  # Shape: [n, d, d]


    #@tf.function
    def algorithm_step(self,mat, kraus, epsilon):
        new_mat = (1-epsilon)*self.apply_dual_channel(tf.linalg.logm((1-epsilon)*self.apply_channel(mat,kraus)+epsilon/tf.cast(self.input_dim,tf.complex128) * tf.eye(self.input_dim,dtype=tf.complex128,batch_shape=[self.parallel_computations])),kraus) + epsilon/tf.cast(self.output_dim, tf.complex128) * tf.eye(self.output_dim,dtype=tf.complex128,batch_shape=[self.parallel_computations])
        return tf.expand_dims(tf.linalg.eigh(new_mat)[1][...,-1],[-1])

    @tf.function
    def current_entropy(self,mat,kraus,epsilon):
        applied = (1-epsilon)*self.apply_channel(mat,kraus)+epsilon/tf.cast(self.input_dim,tf.complex128) * tf.eye(self.input_dim,dtype=tf.complex128,batch_shape=[self.parallel_computations])
        eig = tf.linalg.eigvalsh(applied)
        log_eig = tf.math.log(eig)
        return -tf.reduce_sum(eig * log_eig,axis=[-1])

    @tf.function
    def step(self):
        self.vector.assign(self.algorithm_step(self.get_projectors(self.vector),self.kraus_ops, self.epsilon))




