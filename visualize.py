from classes.Visualizer import *
from classes.Config import *
import math 

channel_id = "RandomUnitary1"

run_path = os.path.join(os.path.sep,"Users", "tdk140", "Desktop", "runs", "RandomUnitary1")
current_path = os.path.dirname(os.path.abspath(__file__))

visualizer_single = Visualizer(config=MinimizerConfig(parent_dir=current_path))
visualizer_tensor= Visualizer(config=MinimizerConfig(parent_dir=run_path))

ent = []
visualizer_single.load_minimizer("RandomUnitary1Single")
for run_dict in visualizer_single.get_runs():
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

visualizer_tensor.load_minimizer(channel_id)

for run_dict in visualizer_tensor.get_runs():
    for run in run_dict.keys():    
        print(visualizer_tensor.get_snapshots(run))

        v = tf.reshape(visualizer_tensor.load_vector(run=run, snapshot=visualizer_tensor.get_snapshots(run)[-1]),[-1])
        d = int(math.sqrt(v.shape[0]))


        c, v1, v2 = schmidt_decomposition(v, d, d)

        print(f"Run {run} max Schmidt coefficient: {c.numpy()[0]}")
