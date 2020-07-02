
import numpy as np
import time, os, sys
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from cellpose import utils
import skimage.filters as filt
from scipy import ndimage
# create matrix of cell probability based on threshold
from skimage.feature import peak_local_max
import skimage.filters as filt
from scipy import ndimage
import json
import time
start= time.time()


img_path = input("Enter the path of images: ")
#img_path = r'./sample_imgs_crop' # the image data shold be 8 bit gray or 24 bit rgb. This should be inside the cellpose root
os.chdir(img_path)
os.chdir("..")


# create paths to save coordinate data
out_root = './result'
coordinate_path = os.path.join(out_root,'coordinate')
mask_path = os.path.join(out_root,'masks')
flow_path = os.path.join(out_root,'flows')
tiles_path = os.path.join(out_root,'tiles')
summary_path = os.path.join(out_root,'summary')
roi_path = os.path.join(out_root,'roi')
styles_path = os.path.join(out_root,'styles')
diams_path = os.path.join(out_root,'diams')
prob_path = os.path.join(out_root,'probabilities')
meta_path = os.path.join(out_root,'meta')

# create the paths
for x in [out_root,coordinate_path,mask_path,flow_path,roi_path,styles_path,diams_path,tiles_path,summary_path,prob_path,meta_path]:
    if not os.path.exists(x):
        os.mkdir(x)

# SET VARIABLES
variable_file = [f for f in os.listdir('./notebooks/') if 'variables.json' in f]
if len(variable_file) != 0:    
    # Read JSON file
    with open(os.path.join('./notebooks',variable_file[0])) as data_file:
        variables = json.load(data_file)
    model_type = variables['model_type']
    channel_type = variables['channel_type']
    diameter = variables['diameter']
    cellprob_threshold = variables['cellprob_threshold']
    flow_threshold = variables['flow_threshold']
    gpu = variables['gpu']
else:
    # vairables for cellpose
    model_type ='cyto' #or nuclei
    channel_type = [0,0] #for gray [0,0], for rgb set [signal channel,nuc channel], r = 1,g=2,b=3
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
    diameter = 20 #set the diameter for the cell, if set False, the program will run an automatic size detection algorithm.
    # threshold used to detect the maxima of cells/coordinate
    # these values are  critical to get good S/N.
    # it would be better to play around with these parameters 
    cellprob_threshold = False
    flow_threshold = False
    # use gpu or not
    gpu = True

# create a img_list
img_list = [f for f in os.listdir(img_path) if '.tif' in f]
imgs = [skimage.io.imread(os.path.join(img_path,f)) for f in img_list]





# RUN CELLPOSE

from cellpose import models

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=gpu , model_type=model_type) # If there is no GPU, this will automatically go to CPU mode.

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# channels = [0,0]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# or if you have different types of channels in each image
channels = [channel_type]*len(img_list)

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended) 
# diameter can be a list or a single number for all images
masks, flows, styles, diams = model.eval(imgs, diameter=diameter, channels=channels,cellprob_threshold= cellprob_threshold,flow_threshold=flow_threshold)


# DISPLAY RESULTS
from cellpose import plot

nimg = len(imgs)
for idx in range(nimg):
    maski = masks[idx]
    flowi = flows[idx][0]

    fig = plt.figure(figsize = (25,15))
    plot.show_segmentation(fig, imgs[idx], maski, flowi, channels=channels[idx])
    plt.savefig(os.path.join(summary_path,img_list[idx].replace('.tif','') +'_summary.png'),format = 'png',dpi = 216,bbox_inches = 'tight')
    plt.close()


## EXTRACT COORDINATE FOR EACH ROI

# reload imgs
imgs = [skimage.io.imread(os.path.join(img_path,f)) for f in img_list]

import pandas as pd
# detecting center of mass for each mask and creating a 2d-array as a roi image
centers = [[list(ndimage.center_of_mass((np.ones(mask.shape)*[mask == k])[0])) for k in np.unique(mask)[1:]] for idx,mask in enumerate(masks)]
dfs = []
rois = []
for c,center in enumerate(centers):
    temp_roi = np.zeros(imgs[c].shape)
    
    if center:
        dfs.append(pd.DataFrame(np.array(center),columns = ['X','Y']))
        temp_roi[np.array(center,dtype = 'int')[:,0],np.array(center,dtype = 'int')[:,1]] = 255
    else:
        dfs.append(pd.DataFrame(np.transpose(np.array([[],[]])),columns = ['X','Y']))
    rois.append(temp_roi)


# save the coordinate files
[df.to_csv(os.path.join(coordinate_path,img_list[idx].replace('.tif','')+'_coordinate.csv'),index = False) for idx,df in enumerate(dfs)]


# SAVE CELL PROABABILITY
# use this cell to check the distribution of cell probability
[skimage.io.imsave(os.path.join(prob_path,img_list[idx].replace('.tif','') +'_prob.tif'),(flow[2] - np.min(flow[2])).astype('uint8')) for idx,flow in enumerate(flows)]

# SAVE CELLPOSE RESULTS
# saveing mask images
[skimage.io.imsave(os.path.join(mask_path,img_list[idx].replace('.tif','') +'_mask.tif'),mask.astype('uint8')) for idx,mask in enumerate(masks)]

# saving flow images
[skimage.io.imsave(os.path.join(flow_path,img_list[idx].replace('.tif','') +'_flow.tif'),flow[0].astype('uint8')) for idx,flow in enumerate(flows)]

# saving cell probability images
[skimage.io.imsave(os.path.join(prob_path,img_list[idx].replace('.tif','') +'_prob.tif'),(flow[2] - np.min(flow[2])).astype('uint8')) for idx,flow in enumerate(flows)]

# saving peak images
[skimage.io.imsave(os.path.join(roi_path,img_list[idx].replace('.tif','') +'_roi.tif'),np.array([imgs[idx],roi]).astype('uint8')) for idx,roi in enumerate(rois)]

end = time.time()
duration = end - start

# create a meta file that stores variables etc.
# Write JSON file
import json
from datetime import datetime

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d%m%Y_%H%M%S")

meta = {"Date":str(now),"Imgs":img_list,"model_type":model_type,"channel_type":channel_type,"diameter":diameter,"cellprob_threshold":cellprob_threshold,"flow_threshold":flow_threshold,"gpu":gpu,"duration":duration}

with open(os.path.join(meta_path,dt_string+'_meta.json'), 'w') as outfile:
    json.dump(meta,outfile)
with open(os.path.join(meta_path,dt_string+'_meta.json')) as data_file:
    meta = json.load(data_file)
