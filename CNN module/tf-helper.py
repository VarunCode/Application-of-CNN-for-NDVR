# all imports
import numpy as np
import math

# TF-IDF Module
def tf_idf_creation(histogram, ds, clusters):
    inv_wcount = defaultdict(int)
    inv_indices = {}
    total_count = 0
    nhist = []
    for arr in range(len(ds)):
        vid = ds[arr]
        for idx, v in enumerate(histogram[vid]):
            if v > 0:
                inv_wcount[idx] += v
                total_count += v
                s = inv_indices.get(idx)
                if s is None:
                    inv_indices[idx] = set([arr])
                else:
                    inv_indices[idx].add(arr)
        arr = np.array(histogram[vid], dtype=np.float32)
        x = sum(arr)
        if x > 0:
            arr = arr / sum(arr)
        if(len(arr) != 0):
            nhist.append(arr)
    for arr,v in inv_wcount.items():
        if v > 0:
            idf[arr] = 1 + math.log(total_count/ v)
        else:
            idf[arr] = 1
    return inv_indices