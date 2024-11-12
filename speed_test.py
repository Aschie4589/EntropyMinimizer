import tensorflow as tf
d = 17
N = 20

# Generate unitary uniformly at random over haar measure.
# To do so, generate random matrix with entries whose real, imaginary parts all are iid random variables
# with standard normal distribution.
with tf.device('/GPU:0'):
    kraus_1 = []
    for i in range(d):
        print(f"Generating random unitaries, step {i+1} of {d}...")
        mat = tf.complex(tf.random.normal([N,N],dtype=tf.float64),tf.random.normal([N,N],dtype=tf.float64))
        unitary = tf.linalg.qr(mat, full_matrices=True)[0]

        kraus_1.append(1/tf.sqrt(tf.cast(d,tf.complex128))*unitary)
        
    kraus_2 = [tf.linalg.adjoint(tf.transpose(el)) for el in kraus_1]
    tensor_kraus = [tf.experimental.numpy.kron(e1, e2) for e1 in kraus_1 for e2 in kraus_2]


    