from sympy.physics.quantum.cg import CG as clebsch_gordan
from ChannelTF import *

class SU2ChannelTF(ChannelTF):
    """
    A class that represents an SU(2) channel embedding applied to a matrix using TensorFlow. Extends ChannelTF.
    
    The channel can be applied to matrices, and it supports saving the embedding tensor 
    to disk and loading it for faster computation in the future. This class also stores the 
    metadata of the saved tensor in a JSON database located in the 'channels' subdirectory.
    """    
    
    def __init__(self):
        super().__init__()

    def initialize(self,J,K,M):
        """
        Initializes the SU2ChannelTF object with quantum numbers J, K, and M. If an embedding
        tensor for the given quantum numbers does not exist, it will be computed and saved.
        
        Args:
            J (float or int): The first quantum number (spin).
            K (float or int): The second quantum number (spin).
            M (float or int): The magnetic quantum number.
        """

        self.id = f"SU2ChannelTF:SU2ChannelTF:SU2_J{J}_K{K}_M{M}"
        try:
            self.load(self.id)
        except (FileNotFoundError,KeyError):
            self.J = J
            self.K = K
            self.M = M 
            self.dimJ = int(2*J+1)
            self.dimK = int(2*K+1)
            self.dimM = int(2*M+1)
            self.input_dim = self.dimJ
            self.output_dim = self.dimK

            self.kraus_ops = tf.stack([self.get_kraus(m) for m in range(self.dimM)])  # Shape: (N, d_out, d_in)
            self.choi_rank = self.kraus_ops.shape[0]
            self.input_dim = self.kraus_ops.shape[2]
            self.output_dim = self.kraus_ops.shape[1]

            self.save()
        return self


    @tf.function
    def get_kraus(self, m):
        """
        Computes the kraus operators from the choi matrix. K_<l|K_m|n>_J = (-1)^(J+n) C(J,-n;K,l|M,m)
        """
        print(f"Initializing the channel (calulating the kraus operator {m+1}/{self.dimM}...)",end=" ")

        kraus = tf.TensorArray(dtype=tf.complex128, size=0, dynamic_size=True, clear_after_read=False)
        for ell in range(self.dimK):
            #k is the row index
            row = [0 for _ in range(self.dimJ)].copy()
            if 0<=self.M+self.J-self.K+ell-m<=self.dimJ-1:
                row[int(self.M+self.J-self.K+ell-m)] = float(clebsch_gordan(self.J,self.M-self.K+ell-m,self.K,self.K-ell,self.M,self.M-m).doit().evalf(50))
            new_row = tf.constant(row, dtype=tf.complex128)
            kraus = kraus.write(ell, new_row)
        print("done!")
        return kraus.stack()

    def get_dual(self):
        dual = SU2ChannelTF()
        dual.id = f"SU2ChannelTF:SU2ChannelTF:SU2_J{self.K}_K{self.J}_M{self.M}"
        dual.J = self.K
        dual.K = self.J
        dual.M = self.M 
        dual.dimJ = int(2*self.K+1)
        dual.dimK = int(2*self.J+1)
        dual.dimM = int(2*self.M+1)

        dual.kraus_ops = tf.transpose(self.kraus_ops,perm=[0,2,1])
        dual.choi_rank = self.kraus_ops.shape[0]
        dual.input_dim = self.kraus_ops.shape[2]
        dual.output_dim = self.kraus_ops.shape[1]
        return dual


    def apply(self, matrix):
        '''
        To apply the channel with the computed kraus operators, we need to transpose the matrix...!
        '''
        return  super().apply(tf.transpose(matrix))


    def load(self,id):

        super().load(id)
        tmp = id.split("_")[1:]
        fJ = float(tmp[0][1:])
        fK = float(tmp[1][1:])
        fM = float(tmp[2][1:])
        self.J = int(fJ) if fJ.is_integer() else fJ
        self.K = int(fK) if fK.is_integer() else fK
        self.M = int(fM) if fM.is_integer() else fM

        self.dimJ = int(2*self.J+1)
        self.dimK = int(2*self.K+1)
        self.dimM = int(2*self.M+1)
