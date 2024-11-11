from sympy.physics.quantum.cg import CG as clebsch_gordan
import numpy as np

class SU2Channel:
    '''
    A channel should be able to be applied to a matrix.

    '''
    def __init__(self, J:float, K:float, M:float):
        self.J = J
        self.K = K
        self.M = M 
        self.dimJ = int(2*J+1)
        self.dimK = int(2*K+1)
        self.dimM = int(2*M+1)

        self.iota = self.get_embedding(self.K,self.J,self.M)

    def get_embedding(self,m, j, k):
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

    def apply(self, matrix):
        return self.dimJ/self.dimK*self.iota.conj().T @ np.kron(matrix, np.eye(self.dimM)) @ self.iota


