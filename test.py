from classes.Minimizer import *
import math

d=6
N=15
epsilon = 1e-9

# Obtain the kraus operators for the random unitary channel
kraus = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.linalg.qr(tf.complex(tf.random.normal([d,N,N],dtype=tf.float64),tf.random.normal([d,N,N],dtype=tf.float64)), full_matrices=True)[0]


# Compute the complementary channel applied to the maximally entangled state
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

# This is the matrix Phi^C(|omega><omega|), where |omega> is the maximally entangled state sqrt(N)^{-1}\sum_i|ii>.
out = complementary_channel(kraus)

# This function takes the output of the channel, and computes the entropy of the corresponding epsilon channel:
# Phi_epsilon(rho) = (1-epsilon)Phi(rho) + epsilon Tr[rho]/dim.
def von_neumann_entropy(rho, epsilon):
    N = rho.shape[-1]
    # Compute eigenvalues of the density matrix
    eigenvalues = tf.linalg.eigvalsh((1 - epsilon) * rho + epsilon / tf.cast(N, tf.complex128) * tf.eye(N, dtype=tf.complex128))
    
    # Ensure eigenvalues are real (small imaginary parts due to numerical errors are removed)
    eigenvalues = tf.math.real(eigenvalues)
       
    # Compute entropy: -sum(lambda * log(lambda))
    entropy = -tf.reduce_sum(eigenvalues * tf.math.log(eigenvalues))
    return entropy

# This function takes care of updating the guess for the entropy of the actual channel, and provides the range and maximum error as well.
def estimate_vn_entropy(rho, epsilon):
    N = rho.shape[-1]
    calc_entropy = von_neumann_entropy(rho,epsilon)
    binentropy = -epsilon*math.log(epsilon)-(1-epsilon)*math.log(1-epsilon)
    estimate = (calc_entropy-epsilon*math.log(N*N)-binentropy/2)/(1-epsilon)
    range = [estimate - binentropy/(2*(1-epsilon)),estimate + binentropy/(2*(1-epsilon))]
    error = binentropy/(2*(1-epsilon))

    return estimate, range, error
                        

est = estimate_vn_entropy(out, 1e-12)
upper_bound_entangled_entropy = est[1][1].numpy()

#print(f"Estimated vn entropy is: {est[0].numpy()}, range is [{est[1][0].numpy()}, {est[1][1].numpy()}]")
print(f"The entropy is for sure lower than: {upper_bound_entangled_entropy}")
print(f"The channel has to do better than this: {upper_bound_entangled_entropy/2}")
print(f"For comparison, this is the value used before: {(1-1/(d*2))*math.log(d)}")

