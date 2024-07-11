'''open_mj2 function to read, bin, and compress files

Rudi Tong, Trennholm Lab'''

from imports import *

import skvideo
skvideo.setFFmpegPath(r'C:\ffmpeg\bin') #(r'C:\Users\erica\anaconda3\Lib\site-packages\skvideo\io') #'C:\ProgramData\Anaconda3\Lib\site-packages\skvideo\io')
import skvideo.io
from skvideo.io import vreader


def load_mj2(fn, start_frame=0, num_frames=0, scale=1, tcompress=1):
    '''Load in mj2 file. Resize and compress in time if necessary.

    fn: file name

    '''
    # Input parameters
    inputparameters = {'-ss': '%s' % (start_frame)}
    # Output parameters
    outputparameters = {'-pix_fmt': 'gray16be'}
    # Calculate total frames after compression
    total_frames = num_frames // tcompress

    # Import video file as numpy.array
    current_frame_size = (1, 1, 1)
    current_frame = np.zeros((current_frame_size))
    vidreader = vreader(fn, inputdict=inputparameters, outputdict=outputparameters, num_frames=num_frames)

    for (i, frame) in tqdm(enumerate(vidreader)):  # vreader is faster than vread, but frame by frame
        if i == 0:
            imagesize = (int(frame.shape[0] * scale), int(frame.shape[1] * scale))
            vid = np.zeros((total_frames, imagesize[0], imagesize[1]))
            print(f"Final video will be shaped {(total_frames, imagesize[0], imagesize[1])}")
            current_frame_size = (tcompress, imagesize[0], imagesize[1])
            current_frame = np.zeros((current_frame_size))

        current_frame[i % tcompress, :, :] = np.squeeze(
            transform.resize(frame, imagesize))  # Resize in place for performance

        if i % tcompress == tcompress - 1:
            vid[i // tcompress, :, :] = current_frame.mean(axis=0)
            current_frame = np.zeros((current_frame_size))

    return vid


v = load_mj2(r'E:\vision_restored\data\EC_RD1_01\20240527\gra\gra_000_100\behav\gra_000_100_eye.mj2', start_frame=0, num_frames=0, scale=1, tcompress=1)

