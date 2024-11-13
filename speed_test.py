from MinimizerTFModule import *
import timeit
from tensorflow.python.client import device_lib 
import logging

# Configure the logging
logging.basicConfig(
    filename='run2.log',  # Log file path
    level=logging.INFO,             # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)


d = 17
N = 30

timing_steps = 1
parallel_computations=100
timing=False

# Generate unitary uniformly at random over haar measure.
# To do so, generate random matrix with entries whose real, imaginary parts all are iid random variables
# with standard normal distribution. The unitary in the QR distribution will be a haar random unitary if the R part has positive diagonal entries (it has in this implementation)
kraus_1 = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.linalg.qr(tf.complex(tf.random.normal([d,N,N],dtype=tf.float64),tf.random.normal([d,N,N],dtype=tf.float64)), full_matrices=True)[0]
kraus_2 = [tf.linalg.adjoint(tf.transpose(el)) for el in kraus_1]
tensor_kraus = [tf.experimental.numpy.kron(e1, e2) for e1 in kraus_1 for e2 in kraus_2]

epsilon = 1/1000
minimizer = MinimizerTFModule(kraus_1,epsilon,parallel_computations)
for i in range(0):
    e = timeit.timeit(minimizer.step,number=timing_steps)
    if timing:
        print(f"Iterated {timing_steps} times. Elapsed time: {e}s")
    print(f"Minimum entropy so far over {parallel_computations} vectors: {tf.reduce_min(tf.cast(tf.abs(minimizer.current_entropy(minimizer.get_projectors(minimizer.vector), minimizer.kraus_ops, minimizer.epsilon)),tf.float64))}")

