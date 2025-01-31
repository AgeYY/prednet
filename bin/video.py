# verify the video straightening hyperthesis with the prednet

import os
import numpy as np
import matplotlib.pyplot as plt

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.agent import Agent
from predusion.curvature import Curvature
from predusion.tools import confidence_interval

from kitti_settings import *

imshape = (128, 160)
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

n_component = 5
video_type = 'natural'
cutoff=2 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end
scale='1x'
ci_bound = [16.5, 83.5] # ci bound of the median across curvatures of different videos
ci_resample = 1000
output_mode = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']

########## Load the video
vsread = VS_reader()
video = vsread.read_video_all_ppd(video_type=video_type, scale=scale) # [number of images in a seq, imshape[0], imshape[1]]

video_ppd = Batch_gen.process_grey_video(video, imshape=imshape) # process the video

##### load the prednet
batch_size = video_ppd.shape[0]

sub = Agent()
sub.read_from_json(json_file, weights_file)

output = sub.output_multiple(video_ppd, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)

########## curverature origin
cv = Curvature()

ct_mean = {}
ct_mean_pca = {}

for key in output:
    output[key] = output[key].reshape(output[key].shape[0], output[key].shape[1], -1) # flatten (n_video, n_frames, n_neurons)
    cv.load_data(output[key])

    ct_mean[key] = cv.curvature_traj(cutoff=cutoff, n_component=None) # ignore the first two video frames due to the pool prediction of the prednet
    ct_mean_pca[key] = cv.curvature_traj(cutoff=cutoff, n_component=n_component) # ignore the first two video frames due to the pool prediction of the prednet

video_flat = video_ppd.reshape((video_ppd.shape[0], video_ppd.shape[1], -1))
cv.load_data(video_flat)
ct_mean_video = cv.curvature_traj(cutoff=cutoff)

ct_mean_video_pca = cv.curvature_traj(cutoff=cutoff, n_component=n_component)
print(ct_mean_video, ct_mean_video_pca)

########## Change of curvature
ct_mean_change = {}
ct_mean_change_pca = {}

ct_mean_change_median = {}
ct_mean_change_pca_median = {}

ct_mean_change_ci = {}
ct_mean_change_pca_ci = {}

for key in ct_mean:
    ct_mean_change[key] = ct_mean[key] - ct_mean_video
    ct_mean_change_median[key] = np.median(ct_mean_change[key])
    ct_mean_change_ci[key] = np.array( confidence_interval(ct_mean_change[key], ci_bound=ci_bound, measure=np.median, n_resamples=ci_resample) )

    ct_mean_change_pca[key] = ct_mean_pca[key] - ct_mean_video_pca
    ct_mean_change_pca_median[key] = np.median(ct_mean_change_pca[key])
    ct_mean_change_pca_ci[key] = np.array( confidence_interval(ct_mean_change_pca[key], ci_bound=ci_bound, measure=np.median, n_resamples=ci_resample) )

#output_mode = 'prediction'
#output = sub.output(video_ppd, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)
#from predusion.ploter import Ploter
#fig, gs = Ploter.plot_seq_prediction(video_ppd[0], output[0])
#plt.show()

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
plt.show()

plt.figure()
plot_curv_layer(ct_mean_change_pca_median, ct_mean_change_pca_ci)
plt.axhline(y = 0, linestyle = '--', linewidth = 1, color = 'black')
plt.legend()
plt.xlabel('Module ID')
plt.ylabel('Change of curvature \n (curvature of neurons - curvature of videos)')
plt.show()
