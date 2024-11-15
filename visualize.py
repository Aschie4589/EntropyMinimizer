from Visualizer import *

visualizer = Visualizer()

visualizer.load_minimizer("CliffordChannel")

#print(*visualizer.get_runs(), sep="\n")

#print(visualizer.get_snapshots("1")[-1])

v = visualizer.load_vector(run="1", snapshot="7585b9da-3909-44b3-b12a-e0ad8f4d4afd")

print(v)


def schmidt_decomposition(vector, dimA, dimB):
    # matrix M satisfies: M_ij = beta_ij, where vector = sum beta_ij e_i(x)e_j
    M = tf.reshape(vector, (dimA, dimB))
    # get svd decomposition of matrix, M = UsV^T
    [sing_val, U, Vt] = tf.linalg.svd(M)
    # Schmidt decomposition is columns of U and Vt (not V transpose because of how tf implements output of svd)
    return (sing_val, U, Vt)

# Example: Quantum state in a flat vector form (size d_A * d_B)
psi = tf.constant([0.707, 0.0, 0.0,0.0,0.0,0.707], dtype=tf.float32)  # Example: Bell state
d_A, d_B = 3, 2  # Dimensions of subsystems A and B

print(schmidt_decomposition(psi, d_A, d_B))
