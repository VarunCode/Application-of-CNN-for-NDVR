import cv2
import operator
import numpy as np

import sys
from scipy.signal import argrelextrema

from myutils import *

def mainextractor(choice, videopath, videoid, dirname, wlen):
    THRESHOLD = False
    THRESHVALUE = 0.7 # TODO hardcoded
    TOPK = False
    LOCALMAXIMA = True
    K = 20

    if choice == 2:
        TOPK = True
        LOCALMAXIMA = False

    wlen = 10

    print("Video :" + videopath)
    # print("Frame dirnameectory: " + dirname)

    cap = cv2.VideoCapture(str(videopath))
    print("Video loaded")

    curr = None
    prev = None

    fdiffs = []
    frames = []
    retval, frame = cap.read()
    i = 1

    while(retval):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr = luv
        if curr is not None and prev is not None:
            diff = cv2.absdiff(curr, prev)
            count = np.sum(diff)
            fdiffs.append(count)
            frame = Frame(i, frame, count)
            frames.append(frame)
        prev = curr
        i = i + 1
        retval, frame = cap.read()

    cap.release()

    shotinfotxt = ""
    print(len(frames))
    if TOPK:
        frames.sort(key=operator.attrgetter("value"), reverse=True)
        ctr = 0
        for keyframe in frames[:K]:
            name = "frame_" + str(keyframe.id) + "_" + videoid
            temp = str(ctr) + "\t" + name + "\t" + videoid + "\t" + "vidname"
            shotinfotxt += temp + "\n"
            ctr += 1
            cv2.imwrite(dirname + name + ".jpg", keyframe.frame)
    elif THRESHOLD:
        for i in range(1, len(frames)):
            if (calcchange(np.float(frames[i - 1].value), np.float(frames[i].value)) >= THRESHVALUE):
                #print("prev:"+str(frames[i-1].value)+"  curr:"+str(frames[i].value))
                name = "frame_" + str(frames[i].id) + "_" + videoid
                temp = str(ctr) + "\t" + name + "\t" + videoid + "\t" + "vidname"
                shotinfotxt += temp + "\n"
                ctr += 1
                cv2.imwrite(dirname + name + ".jpg", frames[i].frame)
    elif LOCALMAXIMA:
        ctr = 0
        diff_array = np.array(fdiffs)
        smoothedarr = smooth(diff_array, wlen)
        fmaxima = np.asarray(argrelextrema(smoothedarr, np.greater))[0]
        for i in fmaxima:
            name = "frame_" + str(frames[i - 1].id) + "_" + videoid
            #print(dirname+name)
            temp = str(ctr) + "\t" + name + "\t" + videoid + "\t" + "vidname"
            shotinfotxt += temp + "\n"
            ctr += 1
            cv2.imwrite(dirname + name + ".jpg", frames[i - 1].frame)

    return shotinfotxt


if __name__ == '__main__':
    # check

    #Video path of the source file
    videopath = sys.argv[1]
    #dirnameectory to store the processed frames
    dirname = sys.argv[2]
    #smoothing window size
    wlen = int(sys.argv[3])
    choice = int(sys.argv[4])


    print mainextractor(choice, videopath, dirname, wlen)
