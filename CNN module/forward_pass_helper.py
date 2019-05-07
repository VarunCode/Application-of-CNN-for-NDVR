import numpy as np
from neural import *

def forward_pass(batch, net):
    conv1, conv2, conv3, conv4, conv5 = forward_pass_alex_net(make_mini_batch, alex_net)
    m1 = np.amax(conv1, axis=(1,2))
    m2 = np.amax(conv2, axis=(1,2))
    m3 = np.amax(conv3, axis=(1,2))
    m4 = np.amax(conv4, axis=(1,2))
    m5 = np.amax(conv5, axis=(1,2))

    res = np.concatenate((m1,m2,m3,m4,m5), axis=1)
    return res