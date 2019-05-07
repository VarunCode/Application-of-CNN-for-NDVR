# all package imports
import numpy as np
import math
import tensorflow as tf
import os
import sys
import pandas as pd
from collections import defaultdict

# ml imports
from random import shuffle
from sklearn.cluster import MiniBatchKMeans as mbkms

# helpers implemented
from neural import *
from video_frames import *
from similarity_module import *
from forward_pass_helper import *
from tf-helper import *
from genhist import *

# Get seed
file = open("../Seed.txt", "res")
seeddata = file.read()
seeddata = seeddata.split("\n")
seeddict = {}
for ele in seeddata:
    ele = ele.split("\t")
    if len(ele) == 2:
        ele[0] = int(ele[0].split("*")[0])
        ele[1] = int(ele[1].split("\res")[0])
        seeddict[ele[0]] = ele[1];
print seeddict

# Params for images and i-means
height,width,dim = 227, 227, 3
batch_size = 1000
clusters = 100
t = 0

# Get query details
querynumber = int(sys.argv[1]);
qv = [seeddict[querynumber]]

sd = defaultdict(int)
id = defaultdict(list)

ap_path = os.path.join("../models/alexnet.npy")
alex_net = np.load(ap_path, encoding='latin1', allow_pickle = True).item()

kd_path = '../keyframes/' - CHECK
md_path = '../kf_shot_info.txt'

kfs = os.listdir(kd_path)

TOTAL_FRAMES = len(kfs)
sample_size = TOTAL_FRAMES

# Parse frame res
seqd = [0 for i in range(TOTAL_FRAMES)]
vidd = defaultdict(list)
i = 0
with open(md_path) as file:
    for line in file:
        sid, kf, vid = line.split('\t')
        if kf + ".jpg" not in kfs:
            continue
        vid = int(vid)
        if len(id[vid]) == 0:
            id[vid] = [0]
        else:
            id[vid].append(0)

        if len(vidd[vid]) == 0:
            vidd[vid] = [kf]
        else:
            vidd[vid].append(kf)

        sd[vid] += 1
        seqd[i] = [vid, kf]
        i += 1
        if(i == TOTAL_FRAMES):
            break

file = [i for i in range(TOTAL_FRAMES)]
shuffle(file)
kframe = file[:sample_size]
rs = np.random.RandomState(0)
kmeans = mkbms(n_clusters=clusters, random_state=rs, init='k-means++')

make_mini_batch = []
for file in kframe:
    if len(seqd[file]) != 2:
        continue
    vid, frame = seqd[file][0], seqd[file][1]
    img = gf(vid, frame)
    if img is None:
        continue
    try:
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
        if vid in qv:
            continue
        make_mini_batch.append(img)
        if len(make_mini_batch) == batch_size:
            t += 1
            make_mini_batch = np.asarray(make_mini_batch)
            res = forward_pass(make_mini_batch, alex_net)
            kmeans.partial_fit(res)
            make_mini_batch = []
            make_mini_batch = np.asarray(make_mini_batch)
            mini_batch_res = []
            for i in range(0,len(make_mini_batch) // 1000):
                cs = make_mini_batch[i*1000:(i+1)*1000]
                res = forward_pass(cs, alex_net)
                if len(mini_batch_res) == 0:
                    mini_batch_res = res
                else:
                    mini_batch_res = np.concatenate((mini_batch_res, res), axis=0)
            kmeans.partial_fit(mini_batch_res)
            make_mini_batch, mini_batch_res = [], []
    except:
        pass

# Create Histogram 
ds = vidd.keys()
vhist = generate_hist(ds, kmeans)
result_set = {}
for video in qv:
    frames = vidd[video]
    images = gvf(video, frames)
    make_mini_batch = np.asarray(images)
    res = forward_pass(make_mini_batch, alex_net)
    out = kmeans.predict(res)
    result_set = query_model(video, inverted_index, clusters)

lfinal = []
for ele in result_set[qv[0]].keys():
    lfinal.append([ds[ele], result_set[qv[0]][ele]])

# Save result_set
df = pd.DataFrame.from_records(lfinal)
df.to_csv("result_sets_original_" + str(querynumber) + ".csv")