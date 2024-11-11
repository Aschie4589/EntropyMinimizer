import tensorflow as tf
import uuid
import os
import json
class ChannelTF:
    '''
    Define a quantum channel using the kraus representation of it. That is, Phi(rho) = Sum_i K_i rho K_i^*, where K_i are the Kraus operators.
    Also allow to save the channel and to restore it from saved file. This requires the channel to have a unique identifier.

    '''
    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.abspath(__file__)) #directory this file is in
        self.channels_json = os.path.join(self.parent_dir, "save","data","channels.json") #path to save data
        self.channels_dir = os.path.join(self.parent_dir, "save","data","channels")

    def initialize(self, kraus_operators:list,id:str=""):
        '''
        Phi= sum K rho K^*
        '''
        self.kraus_ops = tf.stack(kraus_operators)  # Shape: (N, d_out, d_in)
        self.choi_rank = self.kraus_ops.shape[0]
        self.input_dim = self.kraus_ops.shape[2]
        self.output_dim = self.kraus_ops.shape[1]
        if id:
            self.id = f"ChannelTF:ChannelTF:{id}"
        else:
            self.id = f"ChannelTF:ChannelTF:{str(uuid.uuid4())}"

        self.save()
        return self

    def get_dual(self):
        return ChannelTF().initialize(tf.linalg.adjoint(self.kraus_ops))

    def apply(self, matrix):
        rho_expanded = tf.expand_dims(matrix, axis=0)
        transformed= tf.matmul(self.kraus_ops, tf.matmul(rho_expanded, tf.transpose(tf.math.conj(self.kraus_ops), perm=[0, 2, 1])))
        output = tf.reduce_sum(transformed, axis=0)  # Shape: (d', d')
        return output        

    def load(self,id):
        """
        Loads the kraus operators from a file if it exists in the database.
        """
        # Load the JSON database
        if not os.path.exists(self.channels_json):
            raise FileNotFoundError(f"Database not found at {self.channels_json}")
        
        with open(self.channels_json, "r") as f:
            database = json.load(f)
        
        # Search for the tensor by labels J, K, M
        print("Want to load the channel!")
        for entry in database:
            if "id" in entry and entry["id"] == id:
                file_path_rel = entry["file_path"] #relative to dir
                file_path = os.path.join(self.parent_dir, file_path_rel)
                # Read and deserialize the tensor from the file
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Could not find saved tensor at {file_path}!")
                tensor_serialized = tf.io.read_file(file_path)
                self.kraus_ops = tf.io.parse_tensor(tensor_serialized, out_type=tf.complex128)  
                self.id = id
                self.choi_rank = self.kraus_ops.shape[0]
                self.input_dim = self.kraus_ops.shape[2]
                self.output_dim = self.kraus_ops.shape[1]

                print(f"Tensor loaded from {file_path}: {id}")
                return self
        
        # If no match is found
        raise KeyError(f"No channel found with  with id {id} in {self.channels_json}")

    def save(self):
        """
        Saves the computed embedding tensor (iota) to a file and updates the metadata in the JSON database.
        
        The file is saved in the 'channels/data' directory, and the metadata (which includes the 
        quantum numbers and file path) is added to 'channels.json'.
        """
        # Ensure save directory exists
        os.makedirs(self.channels_dir, exist_ok=True)
        
        # Define a unique file name for each tensor based on labels
        file_name = f"kraus-{self.id.split(':')[-1]}.tfrecord"
        file_path_relative = os.path.relpath(os.path.join(self.channels_dir, file_name), self.parent_dir)
        
        # Serialize and save the kraus operators
        tensor_serialized = tf.io.serialize_tensor(self.kraus_ops)
        tf.io.write_file(os.path.join(self.channels_dir, file_name), tensor_serialized)
        
        # Update JSON database
        db_entry = {"id": self.id, "file_path": file_path_relative}
        
        # Load existing database or create a new one
        if os.path.exists(self.channels_json):
            with open(self.channels_json, "r") as f:
                database = json.load(f)
        else:
            database = []
        # Remove entry if already present
        database = [entry for entry in database if not ("id" in entry and entry["id"] == self.id)]

        # Add the new entry
        database.append(db_entry)
        
        # Save the updated database
        with open(self.channels_json, "w") as f:
            json.dump(database, f, indent=4)
        
        print(f"Tensor saved to {file_path_relative} and metadata added to {self.channels_json}")

 