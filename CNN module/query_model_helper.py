import numpy as np
import math
from similarity_module import *
result_set = {}
def query_model(video, inverted_index, clusters):
    query_video = [0 for i in range(clusters)]
    test_video = set()

    for nn in out:
        query_video[nn] += 1
        test_video = test_video | inverted_index[nn]

    query_video = np.array(query_video, dtype=np.float32)
    query_video = query_video / sum(query_video)
    idf_qv = np.multiply(query_video, idf)
    idf = [0 for arr in range(clusters)]

    for tv in test_video:
        idf_doc = np.multiply(normal_vhist[tv], idf)
        cos_sim = cos_similarity(idf_qv, idf_doc)
        if video in result_set:
            result_set[video][tv] = cos_sim
        else:
            result_set[video] = {tv: cos_sim}
    return result_set