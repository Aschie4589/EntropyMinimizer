import tensorflow as tf
import timeit
from collections import deque
import os.path
import uuid
import json 
import importlib
import logging

# Configure the logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),'run.log'),  # Log file path
    level=logging.INFO,             # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)


class EntropyMinimizerTF:

    def __init__(self, dtype = tf.complex128):
        '''
        Entropy minimizer. Will minimize von Neumann entropy by iteratively selecting highest eigenvector of dual_channel(logm(channel(rho))).
        This is guaranteed to decrease entropy at each step.
        Method initialize to initialize.
        '''
        self.parent_dir = os.path.dirname(os.path.abspath(__file__)) #directory this file is in
        self.minimizer_json = os.path.join(self.parent_dir, "save","minimizer.json") #path to minimizer.json
        self.channels_json = os.path.join(self.parent_dir, "save","data","channels.json") #path to save data
        self.channels_dir = os.path.join(self.parent_dir, "save","data","channels")
        self.vectors_dir = os.path.join(self.parent_dir, "save","data","vectors")
        self.snapshots_dir = os.path.join(self.parent_dir, "save","data","vectors","snapshots")
        self.itermax = 20 # nr of iterations to keep track of to see improvements of entropy. Leave 20 or change.
        self.entropy_buffer = deque(maxlen=self.itermax)
        self.dtype = dtype
                

    def initialize(self, channel, epsilon, tolerance=1e15, startvec=None):
        '''
        channel (ChannelTF)                 the channel to run minimization for. channel.apply has to accept a (input_dim,input_dim) tf.tensor as input and return (output_dim, output_tim) tf.tensor as output.
        dual_channel (ChannelTF)            the dual channel. Like channel, but dimensions are swapped. Can be trace preserving but ideally should be unital.
        input_dim (int)                     input dimension for channel
        output_dim (int)                    output dimension for channel
        tolerance (float)                   will stop running the algorithm if entropy decreases by less than this too many times in a row.
        startvec (tensor)                   vector (not projector!) to run the optimization from.
        '''
        self.id = str(uuid.uuid4())
        self.vector_id = str(uuid.uuid4())

        #These are now instances of ChannelTF or SU2ChannelTF. They have an "apply" method, as well as "load" and "save". They also have uuid's.
        self.channel = channel
        self.dual_channel = channel.get_dual()

        self.tolerance = tolerance
        self.epsilon = epsilon

        self.input_dim = self.channel.input_dim
        self.output_dim = self.channel.output_dim

        #These are the functions to apply every time instead
        self.eps_channel = lambda x: (1-epsilon) * self.channel.apply(x) + epsilon/self.output_dim * tf.linalg.trace(x) * tf.eye(self.output_dim,dtype=tf.complex128)
        self.dual_eps_channel = lambda x: (1-epsilon) * self.dual_channel.apply(x) + epsilon/self.input_dim * tf.linalg.trace(x) * tf.eye(self.input_dim,dtype=tf.complex128)

        #initialize the starting vector
        if isinstance(startvec,type(None)):
            self.initialize_random_vector()
        else:   
            self.vector = startvec
        #create the rank one projector
        self.projector = tf.squeeze(tf.tensordot(self.vector, tf.math.conj(self.vector),axes=0))

        self.snapshots = [self.vector]
        return self

    def initialize_random_vector(self):
        '''
        Initializes self.vector to a random vector taken from the uniform density on states.
        '''
        @tf.function
        def wrapper(self,dtype=tf.complex128):
            # Initialize the vector to start optimization with
            #Choosing standard gaussian for both real and imaginary parts gives the uniform distribution in the state space!
            random_real = tf.random.normal(shape=[self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
            random_imaginary = tf.random.normal(shape=[self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
            random_complex = tf.complex(random_real, random_imaginary)
            return tf.cast(random_complex/tf.linalg.norm(random_complex),dtype=dtype)
        self.vector = wrapper(self, dtype=tf.complex128)
        return self

    def iterate_algorithm(self):
        '''
        Run a single step in the minimization. Redefine application of channels for faster (?) tf computation.
        '''
        @tf.function
        def wrapper(proj,kraus1, kraus2, epsilon):
            rho_expanded = tf.expand_dims(proj, axis=0)
            transformed= tf.matmul(kraus1, tf.matmul(rho_expanded, tf.transpose(tf.math.conj(kraus1), perm=[0, 2, 1])))
            ch_applied = tf.reduce_sum(transformed, axis=0)  # Shape: (d', d')
            applied_1 = (1-epsilon)*ch_applied+epsilon/ch_applied.shape[0]*tf.eye(ch_applied.shape[0],dtype=tf.complex128)
            log_applied_1 = tf.linalg.logm(applied_1)

            rho_expanded_out = tf.expand_dims(log_applied_1, axis=0)
            transformed_out= tf.matmul(kraus2, tf.matmul(rho_expanded_out, tf.transpose(tf.math.conj(kraus2), perm=[0, 2, 1])))
            ch_2_applied = tf.reduce_sum(transformed_out, axis=0)  # Shape: (d', d')
            new_mat = (1-epsilon)*ch_2_applied+epsilon/ch_2_applied.shape[0]*tf.eye(ch_2_applied.shape[0],dtype=tf.complex128)

            eigval, eigvec = tf.linalg.eigh(new_mat)
            v = eigvec[:, tf.argmin(tf.abs(eigval))]
            return v,tf.squeeze(tf.tensordot(v, tf.math.conj(v),axes=0))

        

        self.vector,self.projector = wrapper(self.projector, self.channel.kraus_ops,self.dual_channel.kraus_ops,self.epsilon)

        
    def current_entropy(self):
        '''
        Return the current entropy of channel(self.projector)

        '''
        @tf.function
        def wrapper(proj,kraus1, epsilon):
            rho_expanded = tf.expand_dims(proj, axis=0)
            transformed= tf.matmul(kraus1, tf.matmul(rho_expanded, tf.transpose(tf.math.conj(kraus1), perm=[0, 2, 1])))
            ch_applied = tf.reduce_sum(transformed, axis=0)  # Shape: (d', d')
            applied = (1-epsilon)*ch_applied+epsilon/ch_applied.shape[0]*tf.eye(ch_applied.shape[0],dtype=tf.complex128)
            eig = tf.linalg.eigvalsh(applied)
            log_eig = tf.math.log(eig)
            return -tf.reduce_sum(eig * log_eig)
        return wrapper(self.projector,self.channel.kraus_ops, self.epsilon)

    def save(self):
        # Ensure save directory exists
        os.makedirs(self.channels_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)
        os.makedirs(os.path.join(self.snapshots_dir,self.id), exist_ok=True)
        
        # Serialize and save the current state vector
        vector_serialized = tf.io.serialize_tensor(self.vector)
        tf.io.write_file(os.path.join(self.vectors_dir, f"{self.id}_{self.vector_id}_vector.tfrecord"), vector_serialized)

        # Serialize and save the snapshot vectors
        for (i,v) in enumerate(self.snapshots):
            vector_serialized = tf.io.serialize_tensor(v)
            tf.io.write_file(os.path.join(self.snapshots_dir,self.id, f"{self.vector_id}_snapshot_{i}.tfrecord"), vector_serialized)


        # Save the channel
        self.channel.save()
        self.dual_channel.save()

        # Update JSON database
        db_entry = {"minimizer_id": self.id, "channel_id": self.channel.id, "dual_channel_id": self.dual_channel.id, "vector_id":self.vector_id, "epsilon": self.epsilon, "tolerance":self.tolerance, "number_snapshots":len(self.snapshots)}
        
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

    def step_minimization(self,step_number,snapshot_interval=5, verbose=True,log=False):
        '''
        Single step of the iteration. Returns True if minimization has finished.
        '''
        self.iterate_algorithm()
        # Find the new entropy. Append to entropy buffer
        self.entropy_buffer.append(tf.abs(self.current_entropy()))
        if verbose:
            print(f"Entropy so far (iteration {step_number}): {tf.abs(self.entropy_buffer[-1])}", end="\n")
        # Check if we need to stop optimizing
        improvements = [self.entropy_buffer[i] - self.entropy_buffer[i + 1] for i in range(len(self.entropy_buffer) - 1)]
        # Check if we need to save the result.
        if (step_number+1)%snapshot_interval == 0:
            self.snapshots.append(self.vector)
            if log:
                logging.info(f"Snapshot taken. Entropy so far (iteration {step_number}): {tf.abs(self.entropy_buffer[-1])}")

        # We need to stop if they all are small, or if the average is zero or less.

        return len(self.entropy_buffer)==self.itermax and (all([abs(el) < self.tolerance for el in improvements]) or sum(improvements) <= 0)

    def run_minimization(self,log=False,verbose=True):
        '''
        Run the algorithm. Used inside the wrapper minimize_output_entropy() to allow for timing of the process.
        Will stop if the tolerance is reached, or if over the last 20 iterations the entropy has on average stayed the same or not improved.
        '''
        max_iterations = 100000
        print(f"Starting optimization of given channel...")
        if log:
            logging.info(f"Starting optimization of given channel...")
        #Use an entropy buffer. It holds the last iterations_cutoff entropy values.
        self.entropy_buffer.append(tf.abs(self.current_entropy()))

        for i in range(max_iterations):
            if self.step_minimization(i,log=log,verbose=verbose):
                print(f"Finished. Minimal entropy is: {tf.abs(tf.abs(self.entropy_buffer[-1]))} with tolerance {self.tolerance}.")
                if log:
                    logging.info(f"Finished. Minimal entropy is: {tf.abs(tf.abs(self.entropy_buffer[-1]))} with tolerance {self.tolerance}.")
                self.total_iterations = i+1
                return self
    
    def minimize_output_entropy(self, log=False,verbose=True,save=True):
        e = timeit.timeit(lambda:self.run_minimization(log=log,verbose=verbose), number=1)
        print(f"Elapsed time: {e}s. Average s/iteration: {e/self.total_iterations}")
        if save:
            self.save()
