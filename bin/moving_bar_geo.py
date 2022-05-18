# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from predusion.agent import Agent
import predusion.geo_tool as geo_tool
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

nt = 12 # each prediction group contains nt images

out_data_head = 'moving_bar'
output_mode = ['E0', 'E1', 'E2', 'E3']
neural_data_path = 'neural_moving_bar_E' + '.hkl'
#output_mode = ['R0', 'R1', 'R2', 'R3']
#neural_data_path = 'neural_moving_bar_R' + '.hkl'
geo_tool_method = 'cos_xt_xv'
geo_tool_method = 'procrustes_curve_diff_time'
cut0 = 2 # frames from cut_0 to cut
cut = 12
n_com = 120 # used for PCA

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')


## read the pixel
train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)
X_train, label = train_generator.create_all(out_label=True)
speed_list = label
pixel_x = X_train.reshape([X_train.shape[0], X_train.shape[1], -1])

#for i in range(pixel_x.shape[0]):
#    for j in range(pixel_x.shape[1]):
#        pixel_x[i, j] = np.random.permutation(pixel_x[i, j])

for i in range(pixel_x.shape[1]):
    per_idx = np.random.permutation(np.arange(pixel_x.shape[0]))
    #pixel_x[:, i] = np.random.permutation(pixel_x[:, i])
    pixel_x[:, i] = pixel_x[per_idx, i]

pixel_x = np.random.normal(size=pixel_x.shape) # random noise

#from sklearn.decomposition import PCA
# read the neural data
neural_x_all = hkl.load(os.path.join(DATA_DIR, neural_data_path))
neural_x_all['pixel'] = pixel_x # merge pixel_x to neural data

# rearange according to the neural speed
speed_ind = np.argsort(speed_list)
mean_dot, err_dot = [], []

# prednet
for mode in ['pixel'] + output_mode:
    neural_x = neural_x_all[mode].reshape([neural_x_all[mode].shape[0], neural_x_all[mode].shape[1], -1]) # (n_speed, n_time, features)
    neural_x = neural_x[speed_ind][:, cut0:cut]
    neural_x = geo_tool.pca_reduce(neural_x, n_components=n_com)
    if geo_tool_method == 'cos_xt_xv':
        mean_dot_layer, err_dot_layer = geo_tool.cos_xt_xv(neural_x)
    elif geo_tool_method == 'procrustes_curve_diff_time':
        mean_dot_layer, err_dot_layer = geo_tool.procrustes_curve_diff_time(neural_x)
    mean_dot.append(mean_dot_layer)
    err_dot.append(err_dot_layer)

print(mean_dot, err_dot)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(range(-1, 4), mean_dot)
ax.errorbar(range(-1, 4), mean_dot, yerr=err_dot)
ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Layer of the Prednet \n -1 means the pixels')
if geo_tool_method == 'cos_xt_xv':
    ax.set_ylabel('cos of the angle between the tangent \n vector along time and speed')
else:
    ax.set_ylabel('disparity')
plt.show()
