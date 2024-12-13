from classes.Minimizer import *
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

'''
I want to find out what the "best" or "most reasonable" dependence is of the epsilon. i.e.: if I keep track of improvements 
of the algorithm, can I give a reasonable guess as to whether the current search is likely to give better guess than the best
one so far?
'''
# Use random unitary channel. Might need to run tests for other channels as well.
d=10
N=20
log_level=1
channel_name = "random_unitary"
minimization_attempts = 5
# Setup the config for the MinimizerModule
current_path = os.path.dirname(os.path.abspath(__file__))
config = MinimizerConfig(parent_dir=current_path,verbose=True,log=False,save=False, log_level=log_level, log_entropy=1,tolerance=1e-12) # log_entropy=1 means we log the estimated entropy rather than the epsilon entropy...
kraus = 1/tf.sqrt(tf.cast(d, tf.complex128))*tf.linalg.qr(tf.complex(tf.random.normal([d,N,N],dtype=tf.float64),tf.random.normal([d,N,N],dtype=tf.float64)), full_matrices=True)[0]
channel_id = f"{channel_name}_d_{d}"
run_id = f"{channel_id}-1"
minimizer = EntropyMinimizer(config=config)
minimizer.initialize(kraus, id=channel_id, run_id=run_id)
minimizer.message(f"Generated the channel with d={d} unitaries. Performing minimization...")












M = 100


def linear_fit_qr(x, y):
    """
    Perform linear regression using QR decomposition for the case where X is a range.
    
    Parameters:
    - X: The input 1D array of shape (n_samples,)
    - y: The target vector of shape (n_samples,)
    
    Returns:
    - coeff: The coefficients of the linear model (intercept and slope).
    """
    x_arr = np.array(x)
    y_arr = np.array(y)
    # Convert X to a 2D column vector (n_samples, 1)
    X_augmented = np.column_stack([np.ones(x_arr.shape[0]), x_arr])
    
    # Perform QR decomposition on the augmented matrix
    Q, R = np.linalg.qr(X_augmented)
    
    # Compute the coefficients: R^-1 * Q^T * y
    coeff = np.linalg.inv(R) @ Q.T @ y_arr
    
    return coeff


def r_squared(X_list, y_list, coeff):
    """
    Calculate the R-squared (R²) value for the linear regression model.
    
    Parameters:
    - X_list: The input list of shape (n_samples,)
    - y_list: The target list of shape (n_samples,)
    - coeff: The coefficients of the linear model (intercept and slope).
    
    Returns:
    - r2: The R-squared value.
    """
    # Convert input lists to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
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

entropies = []
minimizer.initialize_new_run(run_id = f"{channel_id}-1")
minimizer.message("Starting optimization of given channel...")        
improvements = []
est_entropies = []
finished = False
for _ in range(minimizer.config.max_iterations):
    if not finished:
        finished = minimizer.step_minimization()
        improvement = minimizer.entropy_buffer[-1] - minimizer.entropy_buffer[-2]
        improvements.append(math.log(-improvement))

        # Test if the last M data points give good linear fit. If so, try to predict the entropy that one might get.
        if len(improvements) > M:
            window = improvements[-M:]
            indices = list(range(len(improvements)+1-M,len(improvements)+1))
            coeffs = linear_fit_qr(indices,window)
            rsquared = r_squared(indices, window, coeffs)
            if coeffs[1]<0 and 1-rsquared< 1e-2:
                entropy_est_improvement = math.exp(window[-1]) * math.exp(coeffs[1])/(1-math.exp(coeffs[1]))
                print()
#                print(f"Linear regression gives f(x) = {coeffs[1]}x+{coeffs[0]}")
#                print(f"Estimated overall entropy improvement remaining: {entropy_est_improvement}")
                print(f"Estimated MOE: {minimizer.entropy_buffer[-1]-entropy_est_improvement}")
                print(f"Estimated total number of iterations: {(math.log(minimizer.config.tolerance)-coeffs[0])/coeffs[1]}")

                est_entropies.append([len(improvements),minimizer.entropy_buffer[-1]-entropy_est_improvement])

                # Try this: update M using the estimated number of iterations!
                # Don't do it brutally. Instead, update M to converge to this value? Right now keep 9/10 of old M but add 1/10 of new M.
                M = int(9/10*M+1/10*int(1/6*(math.log(minimizer.config.tolerance)-coeffs[0])/coeffs[1]))
#            print(window)
        # 1. are the last M data points decreasing?

        # 2. Do we have good linear fit? f(x) = a x + b is linear fit.

        # 3. Make prediction of new entropy: new_entropy = current_entropy - exp(last_improvement)* exp(a)/(1-exp(a))

        #print(improvement)

minimizer.message(f"Finished. Minimal entropy is: {minimizer.entropy_buffer[-1]} with tolerance {minimizer.config.tolerance}.")






plt.scatter([e[0] for e in est_entropies], [e[1] for e in est_entropies])
plt.show()



plt.scatter(list(range(len(improvements))),improvements, label="Data")
plt.show()
entropies.append(minimizer.minimizer.lb_entropy.numpy()[0]) # Use the lower bound on the interval of MOE...

MOE_single_channel = min(entropies)
