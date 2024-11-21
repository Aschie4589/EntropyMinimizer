from classes.Minimizer import *
import math

# Generate unitary uniformly at random over haar measure.
# To do so, generate random matrix with entries whose real, imaginary parts all are iid random variables
# with standard normal distribution. The unitary in the QR distribution will be a haar random unitary if the R part has positive diagonal entries (it has in this implementation)
d = 24
N = 29

channel_id = "RandomUnitary1"

#run_path = os.path.join(os.path.sep,"Users", "tdk140", "Desktop", "runs", "entropy")
current_path = os.path.dirname(os.path.abspath(__file__))

config = MinimizerConfig(   parent_dir=current_path,
                            verbose=False,
                            log=True,
                            save=True
                        )


minimizer = EntropyMinimizer(config=MinimizerConfig(parent_dir=os.path.dirname(os.path.abspath(__file__)),log=False, verbose=True, save=False))
try: 
    minimizer.initialize_from_save(channel_id)
    kraus1 = tf.unstack(tf.squeeze(minimizer.minimizer.kraus_ops,axis=1),axis=0)
    print(f"Loaded channel with id {channel_id}")

except FileNotFoundError:
    print(f"Couldn't find a channel with id {channel_id}, will generate a new random channel...")
    kraus1 = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.linalg.qr(tf.complex(tf.random.normal([d,N,N],dtype=tf.float64),tf.random.normal([d,N,N],dtype=tf.float64)), full_matrices=True)[0]
    minimizer.initialize(kraus1, id=channel_id)
    #minimizer.save_kraus()
    print("Done!")

# Generate the tensor channel which is what we will optimize on the server!
kraus2 = [tf.linalg.adjoint(tf.transpose(el)) for el in kraus1]
tkraus = [tf.experimental.numpy.kron(e1, e2) for e1 in kraus1 for e2 in kraus2]

print("Loading tensor channel!")
tensor_minimizer = EntropyMinimizer(config=config)
max_entangled = tf.expand_dims(tf.expand_dims(tf.linalg.normalize(tf.reshape(tf.eye(N, dtype=tf.complex128),[-1]))[0],[1]),[0])

# Set it up 
for i in range(1):
    tensor_minimizer.initialize(tkraus, vector=max_entangled,id=channel_id)
#    if tensor_minimizer.minimizer.entropy.numpy()[0] < (2-1/d)*math.log(d)/2:
#        print("Nope")
#        break
    tensor_minimizer.time_minimization()
#    if tensor_minimizer.minimizer.entropy.numpy()[0] < (2-1/d)*math.log(d)/2:
#        print("Nope nope!")
#        break
