# generate a artificial video based one natural video with predefined curvature
import matplotlib.pyplot as plt
import numpy as np
import os

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.curvature import Curvature, curv_video_neural
from predusion.agent import Agent
from predusion.gen_art_video import Artf_video
from predusion.gen_art_video import gen_artf_video_from_vs_reader
from predusion.tools import confidence_interval

from kitti_settings import *

imshape = (128, 160)
n_video = 10
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

n_component = 5
video_type = 'natural'
cutoff=2 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end. Due to the cutoff, the curvature of artificial video is no longer the same as natural video, but the affect should be minor
scale='1x'
ci_bound = [16.5, 83.5] # ci bound of the median across curvatures of different videos
ci_resample = 1000
output_mode = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
artf_mode = 'fix_norm'

# generate artificial videos
artf_video_batch = gen_artf_video_from_vs_reader(n_video=n_video, mode=artf_mode, n_component_video=n_component) # [n_video, n_frame, *imshape]
video_ppd = Batch_gen.process_grey_video(artf_video_batch, imshape=imshape) # process the video. [n_video, n_frame, *imshape, 3]

##### natural video
#vsread = VS_reader()
#video = vsread.read_video_all_ppd(video_type=video_type, scale=scale) # [number of images in a seq, imshape[0], imshape[1]]
#video_ppd = Batch_gen.process_grey_video(video, imshape=imshape) # process the video

#### save images
#import imageio
#for i, frame in enumerate(artf_video_batch[6]):
#    imageio.imwrite('../figs' + artf_mode + 'frame' + str(i) + '.jpeg', frame)

#for i, frame in enumerate(video[6]):
#    imageio.imwrite('../figs' + artf_mode + 'frame' + str(i) + '.jpeg', frame)

sub = Agent()
sub.read_from_json(json_file, weights_file)

ct_mean_change_median, ct_mean_change_ci, ct_mean_change_pca_median, ct_mean_change_pca_ci = curv_video_neural(video_ppd, sub, output_mode, cutoff, n_component=5)

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
