from classes.Minimizer import *
import math
import argparse

# Generate unitary uniformly at random over haar measure.
# To do so, generate random matrix with entries whose real, imaginary parts all are iid random variables
# with standard normal distribution. The unitary in the QR distribution will be a haar random unitary if the R part has positive diagonal entries (it has in this implementation)

parser = argparse.ArgumentParser(description="Run minimization of random channels to find violation of MOE additivity")
parser.add_argument("--name", required=True, help="Channel name")
parser.add_argument("--N",required=True, help="Size of unitaries to be sampled")
parser.add_argument("--d",required=True, help="Number of unitaries sampled (number of Kraus operators of the channel)")
args = parser.parse_args()
d = int(args.d)
N = int(args.N)

current_path = os.path.dirname(os.path.abspath(__file__))
config = MinimizerConfig(parent_dir=current_path,verbose=False,log=True,save=True)

for i in range(1,1000):
    channel_id = f"{args.name}{i}"
    minimizer = EntropyMinimizer(config=config)
    try: 
        minimizer.initialize_from_save(channel_id)
        kraus1 = tf.unstack(tf.squeeze(minimizer.minimizer.kraus_ops,axis=1),axis=0)
        minimizer.message(f"Loaded channel with id {channel_id}")

    except FileNotFoundError:
        kraus = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.linalg.qr(tf.complex(tf.random.normal([d,N,N],dtype=tf.float64),tf.random.normal([d,N,N],dtype=tf.float64)), full_matrices=True)[0]
        minimizer.initialize(kraus, id=channel_id)
        minimizer.message(f"Couldn't find a channel with id {channel_id}, will generate a new random channel...")
        minimizer.save_kraus()
        minimizer.message("Done!")


    # Set it up 
    violation = False
    for attempt in range(1,101):
        if not violation:
            minimizer.message(f"Starting minimization attempt {attempt}.")
            # initialize a random vector
            minimizer.minimizer.vector.assign(minimizer.minimizer.initialize_random_vectors())

            while not minimizer.step_minimization():
                if minimizer.minimizer.entropy.numpy()[0] < (2-1/d)*math.log(d)/2:
                    minimizer.message(f"Attempt {attempt} at minimization: we have reached the (1-1/2d)log(d) threshold for entropy: {minimizer.minimizer.entropy.numpy()[0]} less than {(2-1/d)*math.log(d)/2}")
                    violation = True
                    break
            minimizer.message(f"Finished attempt {attempt}.")
    if violation:
        minimizer.message(f"Found a vector of low entropy. Violation of additivity of MOE is unlikely!")
    if not violation:
        minimizer.message("Could not find a vector of low entropy. This warrants further investigation!")