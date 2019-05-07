# all imports
import cv2
from scipy import misc

# keyframes data path - CHECK
kd_path = '../mykfs/'
md_path = './myshotinfo.txt'

def gf(video, frame):
    input = kd_path + frame + ".jpg"
    try:
        img = misc.imread(input)
    except Exception as e:
        print e
        return None
    return img


def gvf(video, frames, resize = True):
    input = md_path
    images = []
    for frame in frames:
        try:
            k = input + frame + ".jpg"
            img = misc.imread(k)
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        except Exception as e:
            print e
            continue
    return images