from sympy.physics.quantum.cg import CG as clebsch_gordan
import tensorflow as tf
import os.path
import json

class SU2ChannelTF:
    """
    A class that represents an SU(2) channel embedding applied to a matrix using TensorFlow.
    
    The channel can be applied to matrices, and it supports saving the embedding tensor 
    to disk and loading it for faster computation in the future. This class also stores the 
    metadata of the saved tensor in a JSON database located in the 'channels' subdirectory.
    
    Attributes:
        J (float or int): The first quantum number (spin) used in the channel.
        K (float or int): The second quantum number (spin) used in the channel.
        M (float or int): The third quantum number (magnetic quantum number) used in the channel.
        dimJ (int): The dimension of the Hilbert space associated with J.
        dimK (int): The dimension of the Hilbert space associated with K.
        dimM (int): The dimension of the Hilbert space associated with M.
        eyeM (tf.Tensor): The identity matrix of dimension (2*M + 1).
        iota (tf.Tensor): The embedding tensor representing the SU(2) channel.
        q (tf.Tensor): A tensor calculated from the conjugate transpose of iota.
        db_path (str): Path to the JSON database where tensor metadata is stored.
        save_dir (str): Directory where the embedding tensors will be saved.
    
    Methods:
        __init__(J, K, M): Initializes the SU(2) channel with quantum numbers J, K, M.
        save(): Saves the computed embedding tensor to a file and updates the metadata database.
        load(): Loads the embedding tensor from a file if available in the metadata database.
        get_embedding(m, j, k): Computes and returns the SU(2) channel embedding tensor.
        apply(matrix): Applies the channel to a given matrix using the tensor q and the embedding tensor iota.
    """    
    
    def __init__(self):
        """
        Initializes the SU2ChannelTF object with quantum numbers J, K, and M. If an embedding
        tensor for the given quantum numbers does not exist, it will be computed and saved.
        
        Args:
            J (float or int): The first quantum number (spin).
            K (float or int): The second quantum number (spin).
            M (float or int): The magnetic quantum number.
        """

        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.dir_path, "save","data", "channels.json")
        self.save_dir = os.path.join(self.dir_path, "save","data","channels")

    def initialize(self,J,K,M):
        self.id = f"SU2ChannelTF:SU2ChannelTF:SU2_J{J}_K{K}_M{M}"

        if not self.load(self.id):
            assert isinstance(J, float) or isinstance(J, int)
            assert isinstance(K, float) or isinstance(K, int)
            assert isinstance(M, float) or isinstance(M, int)
            self.J = J
            self.K = K
            self.M = M 
            self.dimJ = int(2*J+1)
            self.dimK = int(2*K+1)
            self.dimM = int(2*M+1)

            self.input_dim = self.dimJ
            self.output_dim = self.dimK
            self.eyeM = tf.eye(self.dimM, dtype=tf.complex128)

            self.iota = self.get_embedding(self.K,self.J,self.M)
            self.q = self.dimJ/self.dimK*tf.transpose(tf.math.conj(self.iota))
            self.save()
        return self

    def save(self):
        """
        Saves the computed embedding tensor (iota) to a file and updates the metadata in the JSON database.
        
        The file is saved in the 'channels/data' directory, and the metadata (which includes the 
        quantum numbers and file path) is added to 'channels.json'.
        """
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Define a unique file name for each tensor based on labels
        file_name = f"tensor_J{self.J}_K{self.K}_M{self.M}.tfrecord"
        file_path_relative = os.path.relpath(os.path.join(self.save_dir, file_name), self.dir_path)
        
        # Serialize and save the tensor
        tensor_serialized = tf.io.serialize_tensor(self.iota)
        tf.io.write_file(os.path.join(self.save_dir, file_name), tensor_serialized)
        
        # Update JSON database
        db_entry = {"id": self.id, "file_path": file_path_relative}
        
        # Load existing database or create a new one
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                database = json.load(f)
        else:
            database = []
        # Remove entry if already present
        database = [entry for entry in database if not (entry["id"] == self.id)]

        # Add the new entry
        database.append(db_entry)
        
        # Save the updated database
        with open(self.db_path, "w") as f:
            json.dump(database, f, indent=4)
        
        print(f"Tensor saved to {file_path_relative} and metadata added to {self.db_path}")

    def load(self,id):
        """
        Loads the embedding tensor (iota) from a file if it exists in the database.
        
        The method checks if the tensor associated with the current J, K, M quantum numbers 
        is already saved. If it exists, the tensor is deserialized and loaded.
        
        Returns:
            bool: True if the tensor was successfully loaded, False otherwise.
        """

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
        self.input_dim = self.dimJ
        self.output_dim = self.dimK


        self.eyeM = tf.eye(self.dimM, dtype=tf.complex128)

        # Load the JSON database
        if not os.path.exists(self.db_path):
            print(f"Database not found at {self.db_path}")
            return False
        
        with open(self.db_path, "r") as f:
            database = json.load(f)
        
        # Search for the tensor by labels J, K, M
        for entry in database:
            if entry["id"] == id:
                file_path_rel = entry["file_path"] #relative to dir
                file_path = os.path.join(self.dir_path, file_path_rel)
                # Read and deserialize the tensor from the file
                if not os.path.exists(file_path):
                    print("Could not find saved tensor... calculating from scratch!")
                    return False
                tensor_serialized = tf.io.read_file(file_path)
                self.iota = tf.io.parse_tensor(tensor_serialized, out_type=tf.complex128)  
                self.q = self.dimJ/self.dimK*tf.transpose(tf.math.conj(self.iota))
                self.id = id

                print(f"Tensor loaded from {file_path}")
                return True
        
        # If no match is found
        print(f"No channel found with  with J={self.J}, K={self.K}, M={self.M} in {self.db_path}")
        return False

    def get_embedding(self, m, j, k):
        """
        Computes the embedding of H_m inside H_j(x)H_k using the Clebsch-Gordan coefficients.
        
        The resulting tensor 'iota' has shape (2*k+1)*(2*j+1) rows and (2*m+1) columns.
        
        Args:
            m (int): The quantum number (spin) associated with H_m.
            j (int): The quantum number (spin) associated with H_j.
            k (int): The quantum number (spin) associated with H_k.
        
        Returns:
            tf.Tensor: The computed embedding tensor.
        """
        @tf.function
        def get_iota():     
            print("Initializing the channel (calulating the embedding...)")

            iota = tf.TensorArray(dtype=tf.complex128, size=0, dynamic_size=True, clear_after_read=False)
            for j1 in range(int(2*j+1)):
                #j1 is spin j-j1
                for k1 in range(int(2*k+1)):    
                    #k1 is spin k-k1
                    print(f"{j1*int(2*k+1)+k1}/{(2*j+1)*(2*k+1)}",end="\r")
                    row = [0 for _ in range(int(2*m+1))].copy()
                    if 0<=m-(j+k-j1-k1)<=2*m:
                        row[int(m-(j+k-j1-k1))] = float(clebsch_gordan(j,j-j1,k,k-k1,m,j-j1+k-k1).doit().evalf(50))
                    new_row = tf.constant(row, dtype=tf.complex128)
                    iota = iota.write(j1*int(2*k+1)+k1, new_row)
            print("Finished!")
            
            return iota.stack()
        return get_iota()

    def apply(self, matrix):
        """
        Applies the channel to a given input matrix.
        
        The matrix is first Kronecker multiplied with the identity matrix of dimension (2*M + 1),
        then multiplied by the iota tensor, and the resulting matrix is returned.
        
        Args:
            matrix (tf.Tensor): The input matrix to which the channel will be applied.
        
        Returns:
            tf.Tensor: The resulting matrix after applying the channel.
        """
        return  tf.matmul(self.q,tf.matmul(tf.experimental.numpy.kron(matrix,self.eyeM),self.iota))

