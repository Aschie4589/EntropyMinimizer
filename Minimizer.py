import tensorflow as tf
import timeit
from collections import deque
import os.path
import uuid
import json
import importlib
import logging
from MinimizerModule import *
from MinimizerConfig import *



class EntropyMinimizer:

    def __init__(self, config:MinimizerConfig=MinimizerConfig()):
        self.config = config


    def initialize(self, kraus, startvec=None):
        self.id = str(uuid.uuid4())
        self.vector_id = str(uuid.uuid4())

        # Instantiate the TF minimizer that will step through the algorithm
        self.minimizer = MinimizerModule(kraus, self.config.epsilon, parallel_computations=self.config.parallel_computations, vec_states = startvec)

        # Create the folder structure needed for saving and logging
        self.ensure_folders([self.config.channels_dir, self.config.log_dir, self.config.snapshots_dir, self.config.vectors_dir])

        # Configure logging for this instance of minimizer
        if self.config.log:
            logging.basicConfig(
                filename=os.path.join(self.config.log_dir,f'{self.id}.log'),  # Log file path
                level=logging.INFO,             # Log level (INFO, DEBUG, ERROR, etc.)
                format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
            )

        # Instantiate the deque that will track the last few entropies calculated. It contains tf tensors!
        self.entropy_buffer = deque(maxlen=self.config.deque_size)
        self.entropy_buffer.append(tf.reduce_min(self.minimizer.entropy))

        # Set the current step to 0.
        self.current_step = 0

        # Snapshots keeps track of vector states throughout the minimization procedure
        self.snapshots = [self.minimizer.vector]
        return self

    def ensure_folders(self, folder_paths):
        for folder in folder_paths:
            os.makedirs(folder, exist_ok=True)        

    '''
    def save_snapshot(self):
        # Serialize and save the snapshot vectors
        for (i,v) in enumerate(self.snapshots):
            vector_serialized = tf.io.serialize_tensor(v)

            tf.io.write_file(os.path.join(self.snapshots_dir,self.id, f"{self.vector_id}_snapshot_{i}.tfrecord"), vector_serialized)

    def save(self):
        # Ensure save directory exists
        os.makedirs(self.channels_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)
        os.makedirs(os.path.join(self.snapshots_dir,self.id), exist_ok=True)
        
        # Serialize and save the current state vector
        vector_serialized = tf.io.serialize_tensor(self.vector)
        tf.io.write_file(os.path.join(self.vectors_dir, f"{self.id}_{self.vector_id}_vector.tfrecord"), vector_serialized)



        # Save the channel
        self.channel.save()
        self.dual_channel.save()

        # Update JSON database
        db_entry = {"minimizer_id": self.id, 
                    "channel_id": self.channel.id, 
                    "dual_channel_id": self.dual_channel.id, 
                    "vector_id":self.vector_id, 
                    "epsilon": self.epsilon, 
                    "tolerance":self.tolerance, 
                    "number_snapshots":len(self.snapshots)}
        
        # Load existing database or create a new one, to append the new entry.
        if os.path.exists(self.minimizer_json):
            with open(self.minimizer_json, "r") as f:
                database = json.load(f)
        else:
            database = []

        # Add the new entry. If duplicate, this is ok because the minimization can be run multiple times and we expect different minimizers.
        # TODO: The minimizer saves snapshots, but these don't have unique names depending on the run! Implement a second id, unique to the run,
        # which is appended to the name of the saved vectors. This would solve this issue.
        database.append(db_entry)
        
        # Save the updated database
        with open(self.minimizer_json, "w") as f:
            json.dump(database, f, indent=4)
        
        print(f"Saved Minimizer with UUID={self.id}. Added metadata to {self.minimizer_json}")
        pass

    def load(self,id):
        # Load the JSON database
        if not os.path.exists(self.minimizer_json):
            raise FileNotFoundError(f"Minimizer database not found at {self.minimizer_json}")
            
        with open(self.minimizer_json, "r") as f:
            minimizer_db = json.load(f)

        # Ensure the id is part of minimizer. Otherwise, throw error!
        for entry in minimizer_db:
            if entry["minimizer_id"] == id:
                self.id = id
                self.vector_id = entry["vector_id"]

                # Load the channels. If not possible, throw error.
                ch_module,ch_class, _ = entry["channel_id"].split(":")
                module = importlib.import_module(ch_module)
                self.channel = getattr(module,ch_class)()
                self.channel.load(entry["channel_id"])

                d_ch_module,d_ch_class, _ = entry["dual_channel_id"].split(":")
                module = importlib.import_module(d_ch_module)
                self.dual_channel = getattr(module,d_ch_class)()
                self.dual_channel.load(entry["dual_channel_id"])

                # Load the vector. If not possible, throw error.
                if not os.path.exists(os.path.join(self.vectors_dir, f"{self.id}_{self.vector_id}_vector.tfrecord")):
                    raise FileNotFoundError("Could not find saved vector state!")
                tensor_serialized = tf.io.read_file(os.path.join(self.vectors_dir, f"{self.id}_{self.vector_id}_vector.tfrecord"))
                self.vector = tf.io.parse_tensor(tensor_serialized, out_type=tf.complex128)  
                #create the rank one projector
                self.projector = tf.squeeze(tf.tensordot(self.vector, tf.math.conj(self.vector),axes=0))

                # Load the snapshots. If not possible, throw error.
                self.snapshots = []
                for i in range(entry["number_snapshots"]):
                    print(i)
                    print(os.path.join(self.snapshots_dir, f"{self.vector_id}_snapshot_{i}.tfrecord"))
                    if not os.path.exists(os.path.join(self.snapshots_dir,self.id, f"{self.vector_id}_snapshot_{i}.tfrecord")):
                        raise FileNotFoundError("Could not find saved vector state snapshots!")
                    tensor_serialized = tf.io.read_file(os.path.join(self.snapshots_dir,self.id, f"{self.vector_id}_snapshot_{i}.tfrecord"))
                    self.snapshots.append(tf.io.parse_tensor(tensor_serialized, out_type=tf.complex128))

                # Load epsilon
                self.epsilon = entry["epsilon"]
                # Load tolerance
                self.tolerance = entry["tolerance"]


                self.input_dim = self.channel.input_dim
                self.output_dim = self.channel.output_dim
                #These are the functions to apply every time instead
                self.eps_channel = lambda x: (1-self.epsilon) * self.channel.apply(x) + self.epsilon/self.output_dim * tf.linalg.trace(x) * tf.eye(self.output_dim,dtype=tf.complex128)
                self.dual_eps_channel = lambda x: (1-self.epsilon) * self.dual_channel.apply(x) + self.epsilon/self.input_dim * tf.linalg.trace(x) * tf.eye(self.input_dim,dtype=tf.complex128)


                return
        
        raise KeyError(f"Did not find a minimizer with id {id} in {self.minimizer_json}")
    '''
    def step_minimization(self):
        '''
        Single step of the iteration. Returns True if minimization has finished.
        '''
        # First step through the algorithm.
        self.minimizer.step()
        self.current_step += 1
        # Find the new minimal entropy across vectors. Append to entropy buffer
        self.entropy_buffer.append(tf.reduce_min(self.minimizer.entropy))

        # Print updates where needed
        if self.config.verbose:
            print(f"Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}", end="\n")
        if self.config.log:
            print(f"Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}", end="\n")


        # Save snapshot if needed.
        if (self.current_step)%self.config.snapshot_interval == 0:
            self.snapshots.append(self.minimizer.vector)
            if self.config.log:
                logging.info(f"Snapshot taken. Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}")

        # Compute improvements. We need to stop if they all are small, or if the average is zero or less.
        improvements = [self.entropy_buffer[i] - self.entropy_buffer[i + 1] for i in range(len(self.entropy_buffer) - 1)]

        return len(self.entropy_buffer)==self.config.deque_size and (all([abs(el) < self.config.tolerance for el in improvements]) or sum(improvements) <= 0)

    def message(self, strg):
        if self.config.log:
            logging.info(strg)
        if self.config.verbose:
            print(strg)

    def run_minimization(self):
        self.message("Starting optimization of given channel...")        
        
        for i in range(self.config.max_iterations):
            #print(self.minimizer.entropy)
            #print(tf.linalg.eigvalsh(self.minimizer.apply_channel(self.minimizer.get_projectors(self.minimizer.vector), self.minimizer.kraus_ops)))
            if self.step_minimization():
                self.message(f"Finished. Minimal entropy is: {self.entropy_buffer[-1]} with tolerance {self.config.tolerance}.")
                return self
    
#    def minimize_output_entropy(self, log=False,verbose=True,save=True):
#        e = timeit.timeit(lambda:self.run_minimization(log=log,verbose=verbose), number=1)
#        print(f"Elapsed time: {e}s. Average s/iteration: {e/self.total_iterations}")
#        if save:
#            self.save()
