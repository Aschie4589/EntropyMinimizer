import tensorflow as tf
import timeit
from collections import deque
import os.path
import uuid
import json 
import importlib

class EntropyMinimizerTF:

    def __init__(self):
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

    def initialize(self, channel, dual_channel, epsilon, tolerance=1e15, startvec=None):
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
        self.dual_channel = dual_channel

        self.tolerance = tolerance
        self.epsilon = epsilon

        self.input_dim = self.channel.input_dim
        self.output_dim = self.channel.output_dim

        #These are the functions to apply every time instead
        self.eps_channel = lambda x: (1-epsilon) * channel.apply(x) + epsilon/self.output_dim * tf.linalg.trace(x) * tf.eye(self.output_dim,dtype=tf.complex128)
        self.dual_eps_channel = lambda x: (1-epsilon) * dual_channel.apply(x) + epsilon/self.input_dim * tf.linalg.trace(x) * tf.eye(self.input_dim,dtype=tf.complex128)

        #initialize the starting vector
        if isinstance(startvec,type(None)):
            self.initialize_random_vector()
        else:   
            self.vector = startvec
        #create the rank one projector
        self.projector = tf.squeeze(tf.tensordot(self.vector, tf.math.conj(self.vector),axes=0))
        return self

    def initialize_random_vector(self):
        '''
        Initializes self.vector to a random vector taken from the uniform density on states.
        '''
        # Initialize the vector to start optimization with
        #Choosing standard gaussian for both real and imaginary parts gives the uniform distribution in the state space!
        random_real = tf.random.normal(shape=[self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        random_imaginary = tf.random.normal(shape=[self.input_dim, 1], mean=0.0, stddev=1.0, dtype=tf.float64)
        random_complex = tf.complex(random_real, random_imaginary)
        self.vector = random_complex / tf.linalg.norm(random_complex)
        return self

    def iterate(self):
        '''
        Run a single step in the minimization
        '''
        new_mat = self.input_dim/self.output_dim*self.dual_eps_channel(tf.linalg.logm(self.eps_channel(self.projector)))
        eigval, eigvec = tf.linalg.eigh(new_mat)
        self.vector = eigvec[:, tf.argmin(tf.abs(eigval))]
        
        self.projector = tf.squeeze(tf.tensordot(self.vector, tf.math.conj(self.vector),axes=0))
        
    def current_entropy(self):
        '''
        Return the current entropy of channel(self.projector)
        '''
        out = self.eps_channel(self.projector)
        eig = tf.linalg.eigvalsh(out)
        log_eig = tf.math.log(eig)
        return -tf.reduce_sum(eig * log_eig)

    def save(self):
        # Ensure save directory exists
        os.makedirs(self.channels_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)
        
        # Serialize and save the current state vector
        vector_serialized = tf.io.serialize_tensor(self.vector)
        tf.io.write_file(os.path.join(self.vectors_dir, f"{self.id}_minimizer_{self.vector_id}.tfrecord"), vector_serialized)

        # Save the channel
        self.channel.save()
        self.dual_channel.save()

        # Update JSON database
        db_entry = {"minimizer_id": self.id, "channel_id": self.channel.id, "dual_channel_id": self.dual_channel.id, "vector_id":self.vector_id, "epsilon": self.epsilon, "tolerance":self.tolerance}
        
        # Load existing database or create a new one, to append the new entry.
        if os.path.exists(self.minimizer_json):
            with open(self.minimizer_json, "r") as f:
                database = json.load(f)
        else:
            database = []

        # Add the new entry. If duplicate, this is ok because the minimization can be run multiple times and we expect different minimizers.
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
                ch_module,ch_class, ch_id = entry["channel_id"].split(":")
                module = importlib.import_module(ch_module)
                self.channel = getattr(module,ch_class)()
                print(type(self.channel))
                if not self.channel.load(ch_id):
                    raise KeyError(f"Did not find a channel with id {entry['channel_id']}")

                d_ch_module,d_ch_class, d_ch_id = entry["dual_channel_id"].split(":")
                module = importlib.import_module(d_ch_module)
                self.dual_channel = getattr(module,d_ch_class)()
                if not self.dual_channel.load(d_ch_id):
                    raise KeyError(f"Did not find a channel with id {entry['dual_channel_id']}")

                # Load the vector. If not possible, throw error.
                if not os.path.exists(os.path.join(self.vectors_dir, f"{self.id}_minimizer_{self.vector_id}.tfrecord")):
                    raise FileNotFoundError("Could not find saved vector state!")
                tensor_serialized = tf.io.read_file(os.path.join(self.vectors_dir, f"{self.id}_minimizer_{self.vector_id}.tfrecord"))
                self.vector = tf.io.parse_tensor(tensor_serialized, out_type=tf.complex128)  

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

    def run_minimization(self):
        '''
        Run the algorithm. Used inside the wrapper minimize_output_entropy() to allow for timing of the process.
        Will stop if the tolerance is reached, or if over the last 20 iterations the entropy has on average stayed the same or not improved.
        '''
        max_iterations = 100000
        iterations_cutoff = 20 #If the new entropy is less than epsilon away from the old one more than this number of times, stop optimizing
        print(f"Starting optimization of given channel...")
        #Define an entropy buffer. It holds the last iterations_cutoff entropy values.
        entropy_buffer = deque(maxlen=iterations_cutoff)
        entropy_buffer.append(tf.abs(self.current_entropy()))

        for i in range(max_iterations):
            self.iterate()
            # Find the new entropy. Append to entropy buffer
            entropy_buffer.append(tf.abs(self.current_entropy()))

            print(f"Entropy so far (iteration {i}): {tf.abs(entropy_buffer[-1])}", end="\n")
            # Check if we need to stop optimizing
            improvements = [entropy_buffer[i] - entropy_buffer[i + 1] for i in range(len(entropy_buffer) - 1)]
            # We need to stop if they all are small, or if the average is zero or less.
            if all([abs(el) < self.tolerance for el in improvements]) or sum(improvements)/len(improvements) <= 0: 
                break

        print(f"Finished. Minimal entropy is: {tf.abs(tf.abs(entropy_buffer[-1]))} with tolerance {self.tolerance}.")
        return self
    
    def minimize_output_entropy(self):
        e = timeit.timeit(self.run_minimization, number=1)
        print(f"Elapsed time: {e}s")
