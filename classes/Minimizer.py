from classes.MinimizerModule import *
from classes.Config import *
import tensorflow as tf
import time
from collections import deque
import os.path
import uuid
import json
import logging

class EntropyMinimizer:

    def __init__(self, config:MinimizerConfig=MinimizerConfig()):
        self.config = config

    def initialize(self, kraus, vector=None, id:str="", run_id:str=""):
        self.id = id if id else str(uuid.uuid4())
        self.run_id = run_id if run_id else str(uuid.uuid4())

        # Instantiate the TF minimizer that will step through the algorithm
        self.minimizer = MinimizerModule(kraus, self.config.epsilon, parallel_computations=self.config.parallel_computations, vec_states = vector)

        # Create the folder structure needed for saving and logging
        self.ensure_folders([self.config.channels_dir, self.config.log_dir, self.config.snapshots_dir, self.config.vectors_dir])

        # Configure logging for this instance of minimizer
        if self.config.log:
            self.logger = self.setup_logger(str(self.id),os.path.join(self.config.log_dir,f'channel_{self.id}.log'))

        # Instantiate the deque that will track the last few entropies calculated. It contains tf tensors!
        self.entropy_buffer = deque(maxlen=self.config.deque_size)
        self.entropy_buffer.append(tf.reduce_min(self.minimizer.entropy))

        # Set the current step to 0.
        self.current_step = 0

        # Snapshots keeps track of vector states throughout the minimization procedure
        self.snapshots = []
        self.snapshots_uuids = []
        # Save the initial configuration.
        if self.config.save:
            self.save_snapshot()
        return self

    def initialize_new_run(self, run_id:str="", vector=None):
        self.run_id = run_id if run_id else str(uuid.uuid4())
        # Re-initialize the TF minimizer vector 
        if vector:
            self.minimizer.vector.assign(vector)
        else:
            self.minimizer.vector.assign(self.minimizer.initialize_random_vectors())
        # Re-initialize the entropy
        self.minimizer.entropy.assign(self.minimizer.current_entropy(self.minimizer.get_projectors(self.minimizer.vector),self.minimizer.kraus_ops,self.minimizer.epsilon))
        # Ensure the folder structure needed for saving and logging
        self.ensure_folders([self.config.channels_dir, self.config.log_dir, self.config.snapshots_dir, self.config.vectors_dir])

        # Re-instantiate the deque that will track the last few entropies calculated. It contains tf tensors!
        self.entropy_buffer = deque(maxlen=self.config.deque_size)
        self.entropy_buffer.append(tf.reduce_min(self.minimizer.entropy))

        # Set the current step to 0.
        self.current_step = 0

        # Reset snapshots
        self.snapshots = []
        self.snapshots_uuids = []
        # Save the initial configuration.
        if self.config.save:
            self.save_snapshot()
        return self


    def setup_logger(self,name, log_file, level=logging.INFO):
        """To setup as many loggers as you want"""
        handler = logging.FileHandler(log_file)        
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger



    def ensure_folders(self, folder_paths):
        for folder in folder_paths:
            os.makedirs(folder, exist_ok=True)        

    def update_database(self):
        # Load existing database or create a new one, to append the new entry.
        if os.path.exists(self.config.minimizer_json):
            with open(self.config.minimizer_json, "r") as f:
                database = json.load(f)
        else:
            database = []
        # Remove the db entry if both the minimizer id and the run id are already present, because we are updating exactly that entry.
        for i in range(len(database)):
            if database[len(database)-1-i]["minimizer_id"] == self.id and database[len(database)-1-i]["run_id"] == self.run_id:
                database.pop(len(database)-1-i)

        # Now create the new minimizer entry in the database.
        db_entry = {
                    "minimizer_id": self.id,
                    "run_id": self.run_id, 
                    "epsilon": self.config.epsilon, 
                    "tolerance": self.config.tolerance, 
                    "snapshots": self.snapshots_uuids,
                    "entropy": tf.abs(self.entropy_buffer[-1]).numpy()
                    }
        
        database.append(db_entry)

        # Save the new database file
        with open(self.config.minimizer_json, "w") as f:
            json.dump(database, f, indent=4)
        
        self.message(f"Saved metadata for minimizer with UUID={self.id} to {self.config.minimizer_json}", log_level=3)

    def save_snapshot(self):
        # Ensure path exists
        self.ensure_folders([os.path.join(self.config.snapshots_dir, self.run_id)])

        # Append the vector to the snapshots
        self.snapshots.append(self.minimizer.vector)

        # Create a UUID for the snapshot
        self.snapshots_uuids.append(str(uuid.uuid4()))

        # Serialize and save the vector to the correct path. Path is ./save/data/vectors/snapshots/{run_id}/{snapshot_uuid}.tfrecord
        vector_serialized = tf.io.serialize_tensor(self.minimizer.vector)
        tf.io.write_file(os.path.join(self.config.snapshots_dir,self.run_id, f"{self.snapshots_uuids[-1]}.tfrecord"), vector_serialized)

        # Update the database
        self.update_database()

    def save_kraus(self):
        '''
        Save Kraus operators to file. To load them, create a new Minimizer (with all the desired config), 
        then initialize it using "initialize_from_save(id)" to load data. This loads the kraus operator
        and calls the initialize method.
        '''
        # This ensures that a record is kept of the kraus operators for this particular instance of the channel and minimizer.
        # To load 
        # Ensure path exists
        self.ensure_folders([self.config.channels_dir])

        # Serialize and save the kraus operators to the correct path. Path is ./save/data/channels/{channel_uuid}.tfrecord
        kraus_serialized = tf.io.serialize_tensor(self.minimizer.kraus_ops)
        tf.io.write_file(os.path.join(self.config.channels_dir, f"{self.id}.tfrecord"), kraus_serialized)

        return self

    def initialize_from_save(self, id, run_id:str="", vector=None):
        '''
        This method tries to load the kraus operators with id "id". If found, they are passed to the initialize method to set everything else up.
        '''

        # Load the operators
        fp = os.path.join(self.config.channels_dir,f"{id}.tfrecord")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Could not find saved kraus operators at {fp}!")
        kraus_serialized = tf.io.read_file(fp)

        # Call the initialize method.
        self.initialize(tf.squeeze(tf.io.parse_tensor(kraus_serialized, out_type=tf.complex128),axis=[1]), id=id, run_id=run_id, vector=vector)
        return self

    def step_minimization(self):
        '''
        Single step of the iteration. Returns True if minimization has finished.
        '''
        # First, step through the algorithm.
        self.minimizer.step()
        self.current_step += 1

        # Choose the correct entropy to log based on config.
        entropies = {0: self.minimizer.entropy,
                      1: self.minimizer.estimated_entropy,
                      2: self.minimizer.ub_entropy,
                      3: self.minimizer.lb_entropy}

        # Find the new minimal entropy across vectors. Append to entropy buffer
        self.entropy_buffer.append(tf.reduce_min(entropies[self.config.log_entropy]))

        # Print updates where needed
        self.message(f"Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}",log_level=1, new_line=False)

        # Save snapshot if needed.
        if self.config.save and (self.current_step)%self.config.snapshot_interval == 0:
            self.save_snapshot()
            self.message(f"Snapshot taken. Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}",log_level=1)

        # Compute improvements. We need to stop if they all are small, or if the average is zero or less.
        improvements = [self.entropy_buffer[i] - self.entropy_buffer[i + 1] for i in range(len(self.entropy_buffer) - 1)]

        return len(self.entropy_buffer)==self.config.deque_size and (all([abs(el) < self.config.tolerance for el in improvements]) or sum(improvements) <= 0)

    def message(self, strg, log_level=0, new_line=True):
        '''
        Log level 0: message is always logged
        Log level x: message is only logged if log level in config is at least x
        '''
        if self.config.log:
            # Log the message if the level of logging in config is at least the log_level specified.
            if log_level <= self.config.log_level:
                self.logger.info(f"[Run {self.run_id}] "+strg)
        if self.config.verbose:
            if new_line:
                print(strg)
            else:
                print(strg, end="\r")

    def run_minimization(self):
        self.message("Starting optimization of given channel...")        
        
        for _ in range(self.config.max_iterations):
            if self.step_minimization():
                self.message(f"Finished. Minimal entropy is: {self.entropy_buffer[-1]} with tolerance {self.config.tolerance}.")
                if self.config.save:
                    self.save_snapshot()
                return self
            
    def time_minimization(self):
        self.message("Starting timed optimization of given channel...")        
        initial_time = time.time()
        for _ in range(self.config.max_iterations):
            it_time = time.time()
            if self.step_minimization():
                self.message(f"Finished. Minimal entropy is: {self.entropy_buffer[-1]} with tolerance {self.config.tolerance}.")
                self.message(f"Total elapsed time: {time.time()-initial_time}s.")
                if self.config.save:
                    self.save_snapshot()
                return self
            self.message(f"Iteration time: {time.time()-it_time}s.", log_level=2)
