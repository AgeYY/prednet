# verify the video straightening hyperthesis with the prednet

import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen

from kitti_settings import *

imshape = (128, 160)
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

n_component = 5
video_id = 6
video_type = 'natural'
cutoff=2 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end
scale='1x'
ci_bound = [16.5, 83.5] # ci bound of the median across curvatures of different videos
ci_resample = 1000
output_mode = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
output_path = './data/natural_video_' + str(video_id) + '.hkl'

########## Load the video
vsread = VS_reader()
video = vsread.read_video_all_ppd(video_type=video_type, scale=scale) # [number of images in a seq, imshape[0], imshape[1]]

video_ppd = Batch_gen.process_grey_video(video, imshape=imshape) # process the video

for im in video_ppd[6]:
    plt.imshow(im.astype(np.ushort))
    plt.show()

walk_video = video_ppd[video_id].astype(np.ushort)
hkl.dump(walk_video, output_path)

load_video = hkl.load(output_path)

for im in load_video:
    plt.imshow(im.astype(np.ushort))
    plt.show()
