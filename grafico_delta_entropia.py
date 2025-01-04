from classes.Minimizer import *
import math
import argparse

'''
For fixed dimension of the Hilbert space N, generate a graph (for as long as it is possible) of Delta S, defined for variable d as follows.

- Construct a random unitary channel with d haar random unitaries. 
- Compute the entropy of the tensor channel Phi(x)Phibar when applied to the maximally entangled state. 
- Compute the minimal output entropy of the single channel Phi, then subtract it twice from the double channel entropy.
- If Delta S is negative, we have (possible) violation of MOE.

We are interested in understanding what size of d would be necessary to (if at all) obtain violation of MOE.
'''
# Set-up argument parsing for command line execution
parser = argparse.ArgumentParser(description="Run minimization of random channels to find violation of MOE additivity")
parser.add_argument("--name", required=False, help="Channel name")
parser.add_argument("--N",required=False, help="Size of unitaries to be sampled", type=int)
parser.add_argument("--dmax",required=False, help="Max number of unitaries sampled (number of Kraus operators of the channel)", type=int)
parser.add_argument('--loglevel', '-l', action='count', default=1, help="Specify number of 'l' to increase log count. Default level:1.")

# Parse the command line inputs
args = parser.parse_args()

if args.dmax:
    dmax = args.dmax
if args.N:
    N = args.N
if args.loglevel:
    log_level = args.loglevel
if args.name:
    channel_name=args.name


# TODO: remove when deploying!
#dmax=10
#N=15
#log_level=1
#channel_name = "random_unitary"


# Setup the config for the MinimizerModule
current_path = os.path.dirname(os.path.abspath(__file__))
config = MinimizerConfig(parent_dir=current_path,verbose=False,log=True,save=False, log_level=log_level, entropy_to_track=1,tolerance=1e-15) # log_entropy=1 means we log the estimated entropy rather than the epsilon entropy...

# We need some helper functions to compute the entropy of the maximally entangled state.

# 1. Compute the complementary channel applied to the maximally entangled state
def complementary_channel(kraus):
    k, N, _ = kraus.shape
    
    # Ensure the tensor is tf.complex128
    kraus = tf.cast(kraus, tf.complex128)

    # Compute U_i U_j^* for all i, j
    Uj_star = tf.linalg.adjoint(kraus)  # Conjugate transpose of each unitary
    product_ij = tf.einsum('ilk,jkm->ijlm', kraus, Uj_star)  # Shape: [k, k, N, N]
    # Compute Tr[U_i U_j^* U_k U_l^*] for all i, j, k, l
    trace_terms = tf.einsum('ijmn,klnm->ijlk', product_ij, product_ij)  # Shape: [k, k, k, k]
    # Final result: Summing the trace terms, scaling and distributing over Kronecker basis
    result = tf.reshape(trace_terms, [k*k, k*k]) / N
    
    return result

# 2. Compute the entropy of rho_epsilon, where rho_epsilon = (1-epsilon)rho+ epsilon Tr[rho]/dim.
def von_neumann_entropy(rho, epsilon):
    N = rho.shape[-1]
    # Compute eigenvalues of the density matrix
    eigenvalues = tf.linalg.eigvalsh((1 - epsilon) * rho + epsilon / tf.cast(N, tf.complex128) * tf.eye(N, dtype=tf.complex128))
    
    # Ensure eigenvalues are real (small imaginary parts due to numerical errors are removed)
    eigenvalues = tf.math.real(eigenvalues)
       
    # Compute entropy: -sum(lambda * log(lambda))
    entropy = -tf.reduce_sum(eigenvalues * tf.math.log(eigenvalues))
    return entropy

# 3. Estimate the entropy of rho from that of rho_epsilon.
def estimate_vn_entropy(rho, epsilon):
    N = rho.shape[-1]
    calc_entropy = von_neumann_entropy(rho,epsilon)
    binentropy = -epsilon*math.log(epsilon)-(1-epsilon)*math.log(1-epsilon)
    estimate = (calc_entropy-epsilon*math.log(N*N)-binentropy/2)/(1-epsilon)
    range = [estimate - binentropy/(2*(1-epsilon)),estimate + binentropy/(2*(1-epsilon))]
    error = binentropy/(2*(1-epsilon))

    return estimate, range, error


# Generate a random unitary. Compute MOE of channel, add DeltaS datapoint and then generate new unitary.
def haar_random_unitary(d):
    z = tf.complex(tf.random.normal([d, d], dtype=tf.float64), tf.random.normal([d, d], dtype=tf.float64))
    q, _ = tf.linalg.qr(z)
    return q


# Setup the kraus operators, starting with just one single unitary.
kraus = tf.expand_dims(haar_random_unitary(N), axis=[0])

DeltaS = []
# Now add new unitaries one at the time.
for d in range(2,dmax+1):
    if d>1:
        #Append one unitary, then rescale so the channel is still TP.
        kraus = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.concat([tf.sqrt(tf.cast(d-1, tf.complex128))*kraus, tf.expand_dims(haar_random_unitary(N),axis=[0])],axis=0)
        #Check that you are TP: print(tf.einsum("ijk,ikl->jl", kraus, tf.linalg.adjoint(kraus))) # Should get identity matrix
    # Calculate Delta= entropy_tensor_channel - 2*MOE_single_channel

    # 1. Get the entropy of the tensor channel
    est_entropy_max_entangled = estimate_vn_entropy(complementary_channel(kraus),1e-9)
    entropy_tensor_channel = tf.squeeze(est_entropy_max_entangled[1][1]).numpy()

    # 2. Get MOE of the single channel
    channel_id = f"{channel_name}_d_{d}"

    minimizer = EntropyMinimizer(config=config)
    minimizer.initialize(kraus, id=channel_id)
    minimizer.message(f"Generated the channel with d={d} unitaries. Performing minimization...")

    minimizer.find_MOE()

    # 4. Append the new delta entropy to the list.
    DeltaS.append((entropy_tensor_channel-2*minimizer.MOE).numpy())
    minimizer.message(f"Current Delta entropies are: {str(DeltaS)}")

#  At the end of everything, save the kraus operators for repeatability.
minimizer.save_kraus()

