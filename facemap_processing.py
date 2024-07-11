from imports import *
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from facemap.pose import pose
from glob import glob

'''
PROCESSING KEYPOINTS

The pose model can be initialized with the following parameters:

    - filenames: (2D-list) List of filenames to be processed.
    - bbox: (list) Bounding box for cropping the video [x1, x2, y1, y2]. If not set, the entire frame is used.
    - bbox_set: (bool) Flag to indicate whether the bounding box has been set. Default is False.
    - resize: (bool) Flag to indicate whether the video needs to be resized.
    - add_padding: (bool) Flag to indicate whether the video needs to be padded. Default is False.
    - gui: (object) GUI object.
    - GUIobject: (object) GUI mainwindow object.
    - net: (object) PyTorch model object.
    - model_name: (str) Name of the model to be used for pose estimation. Default is None which uses the pre-trained model.
'''

#Enter directory path to video:
filepath = 'E:\\vision_restored\\data\\EC_RD1_01\\20240527\\gra\\gra_000_100\\behav\\gra_000_100_eye.mj2'
#File extension must be one of: *.mp4 *.avi *.mj2 *.mkv *.mpeg *.mpg *.asf *m4v
#Save Directory will be created in the input path under Masks

#Run pose estimation on single video

# Initialize pose model.
model = pose.Pose(filenames=[[filepath]])

# process video and save results
model.run()

# Next, use the Facemap GUI to view keypoints tracking on video.

# in terminal...
# conda activate facemap
# python -m facemap