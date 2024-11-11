import tensorflow as tf
from SU2ChannelTF import *
from MinimizerTF import *
from ChannelTF import *

J=20
K=400
i=0

M=K+J-i
channel = SU2ChannelTF().initialize(J,K,M)
print(channel.K)
#dual_channel = SU2ChannelTF().initialize(K,J,M)
epsilon = 1/1000

minimizer = EntropyMinimizerTF().initialize(channel, epsilon=epsilon, tolerance=1e-15)
minimizer.minimize_output_entropy()
minimizer.save()

