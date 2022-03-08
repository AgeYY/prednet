# verify the video straightening hyperthesis with the prednet

import os
import numpy as np
import matplotlib.pyplot as plt

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.agent import Agent
from predusion.tools import curvature

from kitti_settings import *


vsread = VS_reader()

video = vsread.read_video(video_type='natural', video_cate='01', scale='1x') # [number of images in a seq, imshape[0], imshape[1]]

#### process video so that can be fed into the prednet, which should be [number of sequences, number of images in each sequence, imshape[0], imshape[1], 3 channels]
imshape = (128, 160)

video = video[None, ...]

video_ppd = Batch_gen.process_grey_video(video, imshape=imshape)
print(video_ppd.shape)

#for im in video_ppd[0]:
#    plt.imshow(im)
#    plt.show()

##### load the prednet
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
output_mode = 'E3'
batch_size = video_ppd.shape[0]

sub = Agent()
sub.read_from_json(json_file, weights_file)

output = sub.output(video_ppd, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)
output_proc = output.reshape(output.shape[0], output.shape[1], -1) # flatten neural id

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
output_proc_pca = pca.fit_transform(output_proc[0])
print(pca.explained_variance_ratio_)

#ct, ct_mean = curvature(output_proc[0])
ct, ct_mean = curvature(output_proc_pca)
print(np.mean(ct[2:])) # ignore the first two video frames due to the pool prediction of the prednet

video_pca = pca.fit_transform(video.reshape(video.shape[1], -1))
ct_img, ct_mean_img = curvature(video_pca)
print(np.mean(ct_img[2:])) # ignore the first two video frames due to the pool prediction of the prednet

output_mode = 'prediction'
output = sub.output(video_ppd, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)
from predusion.ploter import Ploter
fig, gs = Ploter.plot_seq_prediction(video_ppd[0], output[0])
plt.show()
