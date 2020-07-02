
import numpy as np
import time, os, sys
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from cellpose import utils


# REPLACE FILES WITH YOUR IMAGE PATHS
# files = ['img0.tif', 'img1.tif']\
img_path = input("set the path for images: ")
out_path = input("get out path: ")
filename = input("set some filename: ")


from tensorflow.python.client import device_lib
import tensorflow as tf
sess = tf.Session()
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()
'''
gpu = input("GPU? y/n")
if gpu == 'y':
    gpu = True
elif gpu == 'n':
    gpu = False
'''

files = os.listdir(img_path)
imgs = [skimage.io.imread(os.path.join(img_path,f)) for f in files if ".tif" in f]
nimg = len(imgs)

from cellpose import models

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=get_available_gpus(), model_type='cyto')

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# channels = [0,0]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# or if you have different types of channels in each image
#channels = [[2,3], [0,0], [0,0]]

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended) 
# diameter can be a list or a single number for all images
masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)

import pickle

with open(os.path.join(out_path,filename+"mask.pickle"), "wb") as f:
    pickle.dump(masks, f)

with open(os.path.join(out_path,filename+"flows.pickle"), "wb") as f:
    pickle.dump(flows, f)

with open(os.path.join(out_path,filename+"styles.pickle"), "wb") as f:
    pickle.dump(styles, f)

with open(os.path.join(out_path,filename+"diams.pickle"), "wb") as f:
    pickle.dump(diams, f)

