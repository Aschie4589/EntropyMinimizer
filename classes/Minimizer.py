from classes.MinimizerModule import *
from classes.Config import *
import tensorflow as tf
import time
from collections import deque
import os.path
import uuid
import json
import logging
import numpy as np

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

        # Define the entropy to track
        self.entropy_to_track = {0: self.minimizer.entropy,
                      1: self.minimizer.estimated_entropy,
                      2: self.minimizer.ub_entropy,
                      3: self.minimizer.lb_entropy}[self.config.entropy_to_track]

        # Instantiate the deque that will track the last few entropies calculated. It contains tf tensors!
        self.entropy_buffer = deque(maxlen=self.config.deque_size) 
        self.entropy_buffer.append(tf.reduce_min(self.entropy_to_track))

        # Numpy array of entropy improvements, used to fit exponential convergence and predict final value
        self.ln_entropy_improvements = np.empty((0,))
        self.window_size = self.config.exponential_fit_window_size
        self.predicted_entropy = 0
        self.predicted_steps = -1

        # Set the current step to 0.
        self.current_step = 0

        # Initialize the MOE:
        self.MOE = None

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
        # Re-initialize the entropies
        self.minimizer.update_entropies(self.minimizer.get_projectors(self.minimizer.vector),self.minimizer.kraus_ops)
        # Ensure the folder structure needed for saving and logging
        self.ensure_folders([self.config.channels_dir, self.config.log_dir, self.config.snapshots_dir, self.config.vectors_dir])

        # Re-instantiate the deque that will track the last few entropies calculated. It contains tf tensors!
        self.entropy_buffer = deque(maxlen=self.config.deque_size)
        self.entropy_buffer.append(tf.reduce_min(self.entropy_to_track))

        # Reset the entropy predictions
        self.ln_improvements = np.empty((0,))
        self.window_size = self.config.exponential_fit_window_size
        self.predicted_entropy = 0
        self.predicted_steps = -1

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

        # Find the new minimal entropy across vectors. Append to entropy buffer
        self.entropy_buffer.append(tf.reduce_min(self.entropy_to_track))
        # Calculate the entropy improvement, and append to improvement list
        self.ln_entropy_improvements = np.append(self.ln_entropy_improvements,[tf.math.log(self.entropy_buffer[-2]-self.entropy_buffer[-1]).numpy()]) # positive quantity
        # Print updates where needed
        self.message(f"Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}",log_level=1, new_line=False)

        # Save snapshot if needed.
        if self.config.save and (self.current_step)%self.config.snapshot_interval == 0:
            self.save_snapshot()
            self.message(f"Snapshot taken. Entropy so far (iteration {self.current_step}): {tf.abs(self.entropy_buffer[-1])}",log_level=1)

        # Compute improvements. We need to stop if they all are small, or if the average is zero or less.
        improvements = [self.entropy_buffer[i] - self.entropy_buffer[i + 1] for i in range(len(self.entropy_buffer) - 1)]

        return len(self.entropy_buffer)==self.config.deque_size and (all([abs(el) < self.config.tolerance for el in improvements]) or sum(improvements) <= 0)

    def predict_final_entropy(self):
        def linear_fit_qr(x_arr, y_arr):
            """
            Perform linear regression using QR decomposition for the case where X is a range.
            
            Parameters:
            - X: The input 1D array of shape (n_samples,)
            - y: The target vector of shape (n_samples,)
            
            Returns:
            - coeff: The coefficients of the linear model (intercept and slope).
            """
            # Convert X to a 2D column vector (n_samples, 1)
            X_augmented = np.column_stack([np.ones(x_arr.shape[0]), x_arr])
            
            # Perform QR decomposition on the augmented matrix
            Q, R = np.linalg.qr(X_augmented)
            
            # Compute the coefficients: R^-1 * Q^T * y
            coeff = np.linalg.inv(R) @ Q.T @ y_arr
            
            return coeff
        def r_squared(X, y, coeff):
            """
            Calculate the R-squared (R²) value for the linear regression model.
            
            Parameters:
            - X: The input array of shape (n_samples,)
            - y: The target array of shape (n_samples,)
            - coeff: The coefficients of the linear model (intercept and slope).
            
            Returns:
            - r2: The R-squared value.
            """            
            # Predicted values from the model
            y_pred = coeff[0] + coeff[1] * X
            
            # Total Sum of Squares (TSS)
            y_mean = np.mean(y)
            tss = np.sum((y - y_mean)**2)
            
            # Residual Sum of Squares (RSS)
            rss = np.sum((y - y_pred)**2)
            
            # R-squared
            r2 = 1 - (rss / tss)
            return r2
        l = len(self.ln_entropy_improvements)
        # Check that enough data points have been gathered
        if l >= self.window_size:
            window = self.ln_entropy_improvements[-self.window_size:]
            indices = np.arange(l+1-self.window_size,l+1)
            # Perform linear fit
            coeffs = linear_fit_qr(indices,window)
            rsquared = r_squared(indices, window, coeffs)
            # If fit is good and decreasing:
            if coeffs[1]<0 and rsquared > self.config.exponential_fit_Rsquared_min:
                # Predict entropy
                entropy_est_improvement = tf.math.exp(window[-1]) * tf.math.exp(coeffs[1])/(1-tf.math.exp(coeffs[1]))
                self.predicted_entropy = (self.entropy_buffer[-1]-entropy_est_improvement).numpy()
                # Predict total steps
                self.predicted_steps = int(((tf.math.log(self.config.tolerance)-coeffs[0])/coeffs[1]).numpy())
                # Update window size for more accurate prediction (? this is heuristics right now)
                self.window_size = int(9/10*self.window_size+1/10*1/6*(tf.math.log(self.config.tolerance)-coeffs[0])/coeffs[1])
                return True
            return False

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
                print(strg)

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

    def find_MOE(self):
        self.message(f"Will try to find MOE. Running {self.config.MOE_attempts} minimization attempts.")
        for attempt in range(1,self.config.MOE_attempts+1):
            self.message(f"Initializing minimization attempt {attempt} of {self.config.MOE_attempts}.")
            self.initialize_new_run() # use random id.
            self.message("Starting minimization.")                
            # Perform minimization.
            for _ in range(self.config.max_iterations):
                # Step throuhg the minimization algorithm
                finished = self.step_minimization()
                # Update MOE if necessary
                if not self.MOE or self.MOE > self.entropy_buffer[-1]:
                    self.MOE = self.entropy_buffer[-1]
                self.message(f"Current MOE: {self.MOE}")
                # If there is no significant improvement in the entropy, end the attempt.
                if finished:
                    self.message(f"Algorithm converged. Minimal entropy is: {self.entropy_buffer[-1]} with tolerance {self.config.tolerance}.")
                    if self.config.save:
                        self.save_snapshot()
                    break
                # Make prediction of final entropy
                if self.config.MOE_use_prediction:
                    if self.predict_final_entropy():
                        self.message(f"Predicted final entropy: {self.predicted_entropy} in {self.predicted_steps} iterations.")
                        # If MOE is much smaller than prediction, discard attempt!
                        if self.MOE and self.predicted_entropy - self.MOE > self.config.MOE_prediction_tolerance:
                            self.message(f"Predicted entropy is much larger than current MOE: {self.MOE}, ending the minimization attempt...")
                            break
            self.message(f"Finished attempt {attempt} at minimization. Current MOE: {self.MOE}")
        self.message(f"I have run the algorithm enough times. Final MOE is: {self.MOE}")