import numpy as np
from collections import defaultdict
from forward_pass_helper import *
from sklearn.cluster import MiniBatchKMeans as mbkms

def generate_hist(ds, kmeans):
    vhist = defaultdict(lambda :[0 for i in range(clusters)])
    ds = vidd.keys()
    make_mini_batch, vid_data, seq_data = [], [], []
    i = 0
    for d in ds:
        vid = d
        frames = vidd[d]
        if vid in qv:
            continue
        images = gvf(vid, frames)
        i += 1
        make_mini_batch = np.asarray(images)

        if len(make_mini_batch) == 0:
            continue
        mini_batch_res = []
        res = forward_pass(make_mini_batch, alex_net)
        y = kmeans.predict(res)
        for nearest_neighbour in y:
            vhist[vid][nearest_neighbour] += 1
        make_mini_batch, res = [], []
    return vhist