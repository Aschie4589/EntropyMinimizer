from Minimizer import *

# Generate unitary uniformly at random over haar measure.
# To do so, generate random matrix with entries whose real, imaginary parts all are iid random variables
# with standard normal distribution. The unitary in the QR distribution will be a haar random unitary if the R part has positive diagonal entries (it has in this implementation)
d = 3
N = 6
kraus_1 = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.linalg.qr(tf.complex(tf.random.normal([d,N,N],dtype=tf.float64),tf.random.normal([d,N,N],dtype=tf.float64)), full_matrices=True)[0]
kraus_2 = [tf.linalg.adjoint(tf.transpose(el)) for el in kraus_1]
tensor_kraus = [tf.experimental.numpy.kron(e1, e2) for e1 in kraus_1 for e2 in kraus_2]


# Let's use the Pauli group instead. 
'''
An element is of the form AB where A is in {I, S, H, SH, HS, HSH} and B is in {I,X,Y,Z}
J = 2
dimJ = int(2*J+1)


# Define the matrices of spin, raising and lowering operators for the J representation
Jz = tf.cast(tf.linalg.diag([J-i for i in range(dimJ)]), tf.complex128)

tmp = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
for i in range(dimJ): # i=0 corresponds to m=j-1, i=2j corresponds to m=-j-1 --> J-i-1 is m
    row = [tf.cast(0,tf.float32) for _ in range(dimJ)]
    if i < dimJ-1:
        row[i+1] = tf.cast((J-(J-i-1))*(J+(J-i-1)+1),tf.float32)
    tmp=tmp.write(i, row)       
tmp = tmp.stack()
Jplus = tf.cast(tmp, tf.complex128)

Jminus = tf.linalg.adjoint(Jplus)

Jx = 1/2*(Jplus+Jminus)
Jy = 1/2j*(Jplus-Jminus)


# The set B corresponds (with the physics normalizations) to 1,i*exp(-i*pi*Sx), e^(i pi/2)*exp(-i*pi*Sy), e^(i pi/2)*exp(-i*pi*Sz)... right?
B = []
B.append(tf.eye(dimJ,dtype=tf.complex128))
B.append(1j*tf.linalg.expm(-1j*tf.constant(3.141592653589793,dtype=tf.complex128)*Jx))
B.append(1j*tf.linalg.expm(-1j*tf.constant(3.141592653589793,dtype=tf.complex128)*Jy))
B.append(1j*tf.linalg.expm(-1j*tf.constant(3.141592653589793,dtype=tf.complex128)*Jz))

# The set A includes the Hadamard matrix and the S matrix.

A = []
H = -1j * tf.linalg.expm(1j*tf.constant(3.141592653589793,dtype=tf.complex128)/tf.sqrt(tf.constant(2,dtype=tf.complex128))*(Jx+Jz))
S = tf.math.exp(1j*tf.constant(3.141592653589793,dtype=tf.complex128)/4)*tf.linalg.expm(-1j*tf.constant(3.141592653589793,dtype=tf.complex128)/2*Jz)

A.append(tf.eye(dimJ, dtype=tf.complex128))
A.append(H)
A.append(S)
A.append(tf.matmul(S,H))
A.append(tf.matmul(H,S))
A.append(tf.matmul(H, tf.matmul(S,H)))

#print(H)
#print(S)

kraus1 = [1/tf.sqrt(tf.constant(24,dtype=tf.complex128))*tf.matmul(a,b) for a in A for b in B]
kraus2 = [tf.linalg.adjoint(tf.transpose(el)) for el in kraus1]
tkraus = [tf.experimental.numpy.kron(e1, e2) for e1 in kraus1 for e2 in kraus2]

'''
# Set it up 

config = MinimizerConfig(parallel_computations=10,log=True)
minimizer = EntropyMinimizer(config=config)

minimizer.initialize(tensor_kraus, id="2", run_id="1")
minimizer.time_minimization()
