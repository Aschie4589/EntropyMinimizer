import tensorflow as tf
from SU2ChannelTF import *
from MinimizerTF import *
from ChannelTF import *

J=3
K=3
i=0

M=K+J-i
channel = SU2ChannelTF().initialize(J,K,M)
dual_channel = SU2ChannelTF().initialize(K,J,M)
epsilon = 1/1000

minimizer = EntropyMinimizerTF().initialize(channel, dual_channel,epsilon=epsilon, tolerance=1e-15)
minimizer.minimize_output_entropy()
minimizer.save()

