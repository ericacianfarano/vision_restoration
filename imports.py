import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import os
import os.path
import ast
import random

import scipy
from scipy.io import loadmat
from scipy.io import savemat
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy.stats import pearsonr

from joblib import load, dump
import imageio
import cv2
import random
import math
from types import SimpleNamespace
import colorednoise as cn
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import median_filter
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import median_filter
from scipy import signal
from scipy.io import loadmat


from facemap import process
from glob import glob

from rastermap import Rastermap, utils
from scipy.stats import zscore


from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
import skimage
import skimage.transform as transform
from skimage.measure import block_reduce
import skimage.transform as transform
from skvideo.io import vreader

import pandas as pd
from scipy.optimize import least_squares
from matplotlib.patches import Ellipse
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

#
#
#
# fig, ax = plt.subplots()
# for i in np.arange(0,10000,100):
#     plot_ellipse(ax, x_center[i], y_center[i], width[i], height[i], angle[i], points=poses[i],color = 'red')
# plt.show()
#
# # plotting the center
# plt.figure()
# #plt.scatter(x_center, y_center, c = np.arange(x_center.shape[0]), cmap = 'plasma')
# plt.scatter(x_center, y_center, c = np.append(0, v), cmap = 'plasma')
# plt.show()
#
#
# plt.figure(figsize=(10,4))
# plt.plot(v)
# plt.show()
#
#
#
# plt.figure()
# colors = ['red', 'blue', 'yellow', 'black']
# for i in range(4):
#     plt.scatter(poses[:,i*2], poses[:,(i*2)+1], c = colors[i])
# plt.show()


# plt.figure()
# plt.scatter(x_center, y_center, c = np.arange(x_center.shape[0]), cmap = 'plasma')
# plt.plot(x_center, y_center, c = 'black', alpha = 0.5)
# plt.show()
#
# x_center_smooth = moving_average(x_center, window_size = 3)
# y_center_smooth = moving_average(y_center, window_size = 3)
#
# x_center_smooth_interp = interpolate_nans_1d(x_center_smooth)
# y_center_smooth_interp = interpolate_nans_1d(y_center_smooth)
#
# fps = 20
# frequency_x, fourier_x = analyze_frequency(x_center_smooth_interp, fps)
# frequency_y, fourier_y = analyze_frequency(y_center_smooth_interp, fps)
#
#
# plt.figure(figsize=(12, 9))
# plt.subplot(2, 1, 1)
# plt.plot(frequency_x, fourier_x)
# plt.title('Frequency Analysis of X Velocity')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
#
# plt.subplot(2, 1, 2)
# plt.plot(frequency_y, fourier_y)
# plt.title('Frequency Analysis of Y Velocity')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.show()