# see if the prednet can straighten the flash lag video
# 0. create and save the matrix --> process_kitti.py
# 1. read the data matrix
import matplotlib.pyplot as plt
import hickle as hkl
import numpy as np
import os

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.curvature import Curvature, curv_video_neural
from predusion.agent import Agent
from predusion.gen_art_video import Artf_video
from predusion.gen_art_video import gen_artf_video_from_vs_reader, gen_artf_video_from_nat_video
from predusion.tools import confidence_interval

from kitti_settings import *

output_mode = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
ci_bound = [16.5, 83.5] # ci bound of the median across curvatures of different videos
ci_resample = 1000
cutoff=2 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end. Due to the cutoff, the curvature of artificial video is no longer the same as natural video, but the affect should be minor
n_component = 30
artf_mode = 'fix_end'
verbose = True

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'my_X_test.hkl')

########## Read the video
video_ppd = hkl.load(test_file)[None, ...]
# generate artificial video
#video_ppd = gen_artf_video_from_nat_video(video_ppd, mode=artf_mode, verbose=verbose, n_component_video=n_component) # [n_video, n_frame, *imshape, n_chs]

for artf_video in video_ppd:
    for im in artf_video[:5]:
        plt.imshow(im)
        print('min = {}, max = {}'.format(np.min(im.flatten()), np.max(im.flatten())))
        plt.show()

sub = Agent()
sub.read_from_json(json_file, weights_file)

ct_mean_change_median, ct_mean_change_ci, ct_mean_change_pca_median, ct_mean_change_pca_ci = curv_video_neural(video_ppd, sub, output_mode, cutoff, n_component=n_component)

########## plot out the median
def plot_curv_layer(ct_mean_change_median, ct_mean_change_ci):
    for key in ct_mean_change_median:
        x, y = [int(key[1])], [ct_mean_change_median[key]]
        yerr = (ct_mean_change_ci[key] - y)[..., None]
        yerr = np.abs(yerr)
        if 'E' in key:
            plt.scatter(x, y, c='green')
            plt.errorbar(x, y, yerr=yerr, c='green')
        if 'R' in key:
            plt.scatter(x, y, c='red')
            plt.errorbar(x, y, yerr=yerr, c='red')

plt.figure()
plot_curv_layer(ct_mean_change_median, ct_mean_change_ci)
plt.axhline(y = 0, linestyle = '--', linewidth = 1, color = 'black')
plt.legend()
plt.xlabel('Module ID')
plt.ylabel('Change of curvature \n (curvature of neurons - curvature of videos)')
plt.ylim([-0.3, 1.4])
plt.show()

plt.figure()
plot_curv_layer(ct_mean_change_pca_median, ct_mean_change_pca_ci)
plt.axhline(y = 0, linestyle = '--', linewidth = 1, color = 'black')
plt.legend()
plt.xlabel('Module ID')
plt.ylabel('Change of curvature \n (curvature of neurons - curvature of videos)')
plt.ylim([-0.3, 1.4])
plt.show()
