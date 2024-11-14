import tensorflow as tf
from SU2ChannelTF import *
from MinimizerTF import *
from ChannelTF import *
import logging

# Configure the logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),'run.log'),  # Log file path
    level=logging.INFO,             # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

#J=5
#K=5
#i=0

#M=K+J-i
#channel = SU2ChannelTF().initialize(J,K,M)
#epsilon = 1/1000

#minimizer = EntropyMinimizerTF()
#minimizer = EntropyMinimizerTF().initialize(channel, epsilon=epsilon, tolerance=1e-15)
#minimizer.minimize_output_entropy()
#print(minimizer.snapshots)

d = 17
N = 20

# Generate unitary uniformly at random over haar measure.
# To do so, generate random matrix with entries whose real, imaginary parts all are iid random variables
# with standard normal distribution.
with tf.device('/GPU:0'):
    kraus_random_unitary = []
    for i in range(d):
        logging.info(f"Generating random unitaries, step {i+1} of {d}...")
        mat = tf.complex(tf.random.normal([N,N],dtype=tf.float64),tf.random.normal([N,N],dtype=tf.float64))
        unitary = tf.linalg.qr(mat, full_matrices=True)[0]

        kraus_random_unitary.append(1/tf.sqrt(tf.cast(d,tf.complex128))*unitary)
        
    kraus2 = [tf.linalg.adjoint(tf.transpose(el)) for el in kraus_random_unitary]
    channel1 = ChannelTF(dtype=tf.complex128)
    channel1.initialize(kraus_random_unitary)

    #channel2 = ChannelTF()
    #channel2.initialize(kraus2)

    tensor_channel = ChannelTF(dtype=tf.complex128)
    tensor_kraus = [tf.experimental.numpy.kron(e1, e2) for e1 in kraus_random_unitary for e2 in kraus2]
    tensor_channel.initialize(tensor_kraus)


    logging.info("Minimizing entropy of channels separately...")
    # Note: total entropy is just twice moe of one channel from def of this channel
    epsilon = 1/1000

    MOEs = []
    for i in range(0):
        logging.info(f"Minimization of single channel, iteration {i+1} of {100}")
        minimizer1 = EntropyMinimizerTF(dtype=tf.complex128)
        minimizer1 = EntropyMinimizerTF().initialize(channel1, epsilon=epsilon, tolerance=1e-15)

        minimizer1.run_minimization(log=False,verbose=True)
        MOEs.append(2*tf.abs(minimizer1.current_entropy()))

    logging.info("Now running the combined channel...")
    tensor_MOEs = []
    for i in range(10):
        tensor_minimizer = EntropyMinimizerTF(dtype=tf.complex128)
        tensor_minimizer.initialize(tensor_channel, epsilon=epsilon,tolerance=1e-15)
        tensor_minimizer.run_minimization(log=False,verbose=True)
        tensor_MOEs.append(tf.abs(tensor_minimizer.current_entropy()))

    logging.info("Summary.")
    logging.info(f"Minimal entropies for channels separately: {str(MOEs)}")
    logging.info(f"Minimal entropies for TP: {str(tensor_MOEs)}")

    if min(MOEs) > min(tensor_MOEs):
        print("Possible violation of additivity of MOE found!")
        print(f"Minimzer id 1: {minimizer1.id}")
        print(f"Minimzer id 3: {tensor_minimizer.id}")
        print(MOEs)
        print(tensor_MOEs)
        logging.info("Possible violation of additivity of MOE found!")
        logging.info(f"Minimzer id 1: {minimizer1.id}")
        logging.info(f"Minimzer id 3: {tensor_minimizer.id}")



