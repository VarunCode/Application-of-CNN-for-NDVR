from mykfextractor import *
import os
import sys

videospath = sys.argv[1]
kfspath = sys.argv[2]

vidslist = os.listdir(videospath)
shotinfotxt = ""


vnamesdict = {}
f = open("Video_Complete.txt", "r")
data = f.read()
data = data.split("\n")
for ele in data:
    ele = ele.split("\t")
    if(len(ele) == 13):
        vnamesdict[ele[3][1:-1]] = ele[0]

for vid in vidslist:
    vidid = vnamesdict[vid]
    try:
        shotinfotxt += mainextractor(2, videospath + vid, vidid, kfspath, 10)
    except:
        print "Error for", vidid
        pass

f = open("myshotinfo.txt", "w")
f.write(shotinfotxt)
f.close()
