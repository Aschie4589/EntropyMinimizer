import numpy as np
from scipy.linalg import logm

class EntropyMinimizer:

    def __init__(self, channel, dual_channel, input_dim, output_dim, tolerance, startvec=None):
        self.channel = channel
        self.dual_channel = dual_channel
        self.tolerance = tolerance
        self.input_dim = input_dim
        self.output_dim = output_dim
        #initialize the starting vector
        if isinstance(startvec,type(None)):
            self.initialize_random_vector()
        else:   
            self.vector = startvec
        #create the rank one projector
        self.projector = self.vector@np.conj(self.vector.T)
        pass

    def initialize_random_vector(self):
        # Initialize the vector to start optimization with
            #Choosing standard gaussian for both real and imaginary parts gives the uniform distribution in the state space!
        self.vector = np.random.randn(self.input_dim, 1) +1j * np.random.randn(self.input_dim, 1) 
        self.vector = self.vector / np.linalg.norm(self.vector)
        return self

    def iterate(self):
        new_mat = self.input_dim/self.output_dim*self.dual_channel(logm(self.channel(self.projector)))
        eval, evec = np.linalg.eigh(new_mat)
        self.vector = np.matrix(evec[:,-1]).T
        self.projector = self.vector@np.conj(self.vector.T)
        pass

    def current_entropy(self):
        out = self.channel(self.projector)
        return np.real_if_close(np.trace(-logm(out)@out))

    def minimize_output_entropy(self):
        max_iterations = 10000
        iterations_cutoff = 20 #If the new entropy is less than epsilon away from the old one more than this number of times, stop optimizing
        print(f"Starting optimization of given channel...")
        # Initialize the vector to start optimization with
        #Optimize
        current_entropy = self.current_entropy()
        new_entropy = 0
        current_cutoff = 0
        print(current_entropy)

        for i in range(max_iterations):
            self.iterate()
            # Find the new entropy.
            new_entropy = self.current_entropy()

            print(f"Entropy so far (iteration {i}): {new_entropy}", end="\n")
            # Check if we need to stop optimizing
            if np.abs(current_entropy-new_entropy)<self.tolerance:
                current_cutoff += 1
            else:
                current_cutoff = 0

            if current_cutoff>= iterations_cutoff:
                # We have reached the desired precision
                break
            current_entropy = new_entropy

        print(f"Finished. Minimal entropy is: {new_entropy} with tolerance {self.tolerance}.")
        return self
    