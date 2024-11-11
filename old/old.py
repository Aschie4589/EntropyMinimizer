
def get_embedding(m, j, k):
    """
    Get the embedding of H_m inside H_j(x)H_k
    Assuming basis to be ordered lexicographically - |jk> is element 1, |j,k-1> is 2, then all the way to |-j,-k>.
    Iota has to have 2m+1 columns and (2k+1)(2j+1) rows.
    """
    iota = np.zeros((int(2*j+1)*int(2*k+1), int(2*m+1)))
    for m1 in range(int(2*m+1)):
        # m1 corresponds to spin m-m1
        for j1 in range(int(2*j+1)):
            #j1 is spin j-j1
            for k1 in range(int(2*k+1)):    
                #k1 is spin k-k1
                iota[j1*int(2*k+1)+k1, m1] = clebsch_gordan(j,k,m,j-j1,k-k1,m-m1)
        
    return iota

mem = {}
def apply_phi_channel(J,K,M, rho):
    dimJ = int(2*J+1)
    dimK = int(2*K+1)
    dimM = int(2*M+1)
    # memorize past embeddings to speed up calculations.
    if not(dimJ in mem):
        mem[dimJ] = {dimK: {dimM: get_embedding(K,J,M)}}
    elif not(dimK in mem[dimJ]):
        mem[dimJ][dimK] = {dimM: get_embedding(K,J,M)}
    elif not(dimM in mem[dimJ][dimK]):
        mem[dimJ][dimK][dimM] = get_embedding(K,J,M)
    iota = mem[dimJ][dimK][dimM]
    return dimJ/dimK*iota.conj().T @ np.kron(rho, np.eye(dimM)) @ iota

def apply_phi_epsilon_channel(J,K,M,epsilon,rho):
    return (1-epsilon) * apply_phi_channel(J,K,M,rho) + epsilon/(2*K+1) * np.trace(rho) * np.eye(int(2*K+1))

def minimize_output_entropy(J,K,M,epsilon,tolerance=1e-12,startvec = None, real_time_visualize = False):
    max_iterations = 10000
    iterations_cutoff = 20 #If the new entropy is less than epsilon away from the old one more than this number of times, stop optimizing
    print(f"Starting optimization of output entropy of channel Phi^{M}_({J},{K})...")
    # Initialize the vector to start optimization with
    if isinstance(startvec,type(None)):
        #Choosing standard gaussian for both real and imaginary parts gives the uniform distribution in the state space!
        startvec = np.random.randn(int(2*J)+1, 1) +1j * np.random.randn(int(2*J)+1, 1) 
        startvec = startvec / np.linalg.norm(startvec)
    projector = startvec@np.conj(startvec.T)
    # Initialize plot if needed
    if real_time_visualize:
        visualizer.initialize_plot()
    #Optimize
    current_entropy = np.real_if_close(np.trace(-logm(apply_phi_channel(J,K,M, projector))@apply_phi_channel(J,K,M,projector)))
    new_entropy = 0
    current_cutoff = 0

    for i in range(max_iterations):
        # Calculate Phi^*(log(Phi(rho)))
        output = (2*J+1)/(2*K+1)*apply_phi_epsilon_channel(K,J,M,epsilon,logm(apply_phi_epsilon_channel(J,K,M,epsilon, projector)))
        # Diagonalize
        eigvals, eigvecs = np.linalg.eigh(output)
        # The new rho is the projection onto the highest eigenvalue (here index -1)
        vec = np.matrix(eigvecs[:,-1])
        projector = np.conj(vec.T)@vec
        # Update the visualization if needed!
        if real_time_visualize:# and i % 2 == 0:
            visualizer.update_star_plot(vec, J, i)

        # Find the new entropy.
        new_entropy = np.real_if_close(np.trace(-logm(apply_phi_channel(J,K,M, projector))@apply_phi_channel(J,K,M,projector)))

        print(f"Entropy so far (iteration {i}): {new_entropy}", end="\r")
        # Check if we need to stop optimizing
        if np.abs(current_entropy-new_entropy)<tolerance:
            current_cutoff += 1
        else:
            current_cutoff = 0

        if current_cutoff>= iterations_cutoff:
            # We have reached the desired precision
            break
        current_entropy = new_entropy

    print(f"Finished. Minimal entropy is: {new_entropy} with tolerance {tolerance}.")
    return vec
