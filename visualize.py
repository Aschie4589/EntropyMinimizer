from Visualizer import *
import math 
from matplotlib import pyplot as plt

visualizer = Visualizer()

ent = []
visualizer.load_minimizer("CliffordChannelLowDim")
for run_dict in visualizer.get_runs():
    for run in run_dict.keys():
        ent.append(run_dict[run]) 

bins = []
count = []
for e in ent:
    found = False
    for (i,b) in enumerate(bins):
        if abs(e-b)<1e-7:
            count[i] +=1
            found = True
    if not found:
        bins.append(e)
        count.append(1)



print(*sorted(list(zip(bins,count))), sep="\n")
    


#print(*visualizer.get_runs(), sep="\n")

def schmidt_decomposition(vector, dimA, dimB):
    # matrix M satisfies: M_ij = beta_ij, where vector = sum beta_ij e_i(x)e_j
    M = tf.reshape(vector, (dimA, dimB))
    # get svd decomposition of matrix, M = UsV^T
    [sing_val, U, Vt] = tf.linalg.svd(M)
    # Schmidt decomposition is columns of U and Vt (not V transpose because of how tf implements output of svd)
    return (sing_val, U, Vt)


for run_dict in visualizer.get_runs():
    for run in run_dict.keys():    
        v = tf.reshape(visualizer.load_vector(run=run, snapshot=visualizer.get_snapshots(run)[-1]),[-1])
        d = int(math.sqrt(v.shape[0]))


        c, v1, v2 = schmidt_decomposition(v, d, d)

        print(f"Run {run} max Schmidt coefficient: {c.numpy()[0]}")
