
import os, argparse
from datetime import datetime
import numpy as np
from timeit import default_timer as timer
import skvideo.io
import skimage.transform as transform
from skimage.morphology import skeletonize
from skimage import exposure
from scipy.io import loadmat
try:
    import scipy.fft as fft
except:
    import scipy.fftpack as fft
import scipy.signal as signal
import glob
import matplotlib.pyplot as pl
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing, binary_dilation, label, \
    laplace
from configparser import ConfigParser


def load_data(fn, start_frame=0, end_frame=0, resize=None):
    '''Load the video file specified in path.
       File should be in .mj2 format.
       Uses ffmpeg backend and scikit-video to load video file.

       INPUT
       fn           : Filepath to video file (.mj2 format)
       start_frame  : First frame to read (count from 0)
       end_frame    : Last frame to read
       resize       : Factor to resize image to. Skip if None

       OUTPUT
       vid          : Video file as numpy.array (frames x dim1 x dim2)
    '''

    # Create input and output parameters for ffmpeg
    inputparameters = {'-ss': '%s' % (start_frame)}  # -ss seeks to the position/frame in video file
    outputparameters = {'-pix_fmt': 'gray16be'}  # specify to import as uint16, otherwise it's uint8
    if end_frame == 0:
        num_frames = 0
    else:
        num_frames = end_frame - start_frame

    # Import video file as numpy.array
    vidreader = skvideo.io.vreader(fn, inputdict=inputparameters, outputdict=outputparameters, num_frames=end_frame)
    for (i, frame) in enumerate(vidreader):  # vreader is faster than vread, but frame by frame
        if i == 0:
            imagesize = (int(frame.shape[0] * resize), int(frame.shape[1] * resize))
            vid = np.zeros((end_frame, imagesize[0], imagesize[1]))

        vid[i, :, :] = np.squeeze(transform.resize(frame, imagesize))  # Resize in place for performance

    return vid



