import tensorflow as tf
from SU2ChannelTF import *
from MinimizerTF import *
from ChannelTF import *

J=5
K=20
i=0

M=K+J-i
channel = SU2ChannelTF().initialize(J,K,M)
epsilon = 1/1000

minimizer = EntropyMinimizerTF()
minimizer = EntropyMinimizerTF().initialize(channel, epsilon=epsilon, tolerance=1e-15)
minimizer.minimize_output_entropy()
print(minimizer.snapshots)
