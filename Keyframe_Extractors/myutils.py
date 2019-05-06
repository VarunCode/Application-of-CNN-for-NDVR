import cv2
import operator
import numpy as np

import sys
from scipy.signal import argrelextrema

# smooth func
def mysmoothfunc(x, wlen=13, window='hanning'):
    s = np.r_[2 * x[0] - x[wlen : 1 : -1], x, 2 * x[-1] - x[-1 : -wlen : -1]]
    # print s

    if window == 'flat':
        w = np.ones(wlen, 'd')
    else:
        w = getattr(np, window)(wlen)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[wlen - 1 : -wlen + 1]

# Frame class
class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

def calcchange(a, b):
   x = (b - a) / max(a, b)
   print x
   return x
