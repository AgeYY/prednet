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
#output_mode = ['E0', 'E1', 'E2', 'E3']
#neural_data_path = 'neural_moving_bar_E' + '.hkl'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_moving_bar_R' + '.hkl'
geo_tool_method = 'procrustes_curve_diff_time'
geo_tool_method = 'dim_manifold'
geo_tool_method = 'cos_xt_xv'
cut0 = 2 # frames from cut_0 to cut
cut = 12
n_com_procrustes = 4
n_com_cos = 20

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
print_message = False


## read the pixel
train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)
X_train, label = train_generator.create_all(out_label=True)
speed_list = label
pixel_x = X_train.reshape([X_train.shape[0], X_train.shape[1], -1])

def shuffle_neuron(pixel_x):
    pixel_x_shuffle = np.empty(pixel_x.shape)
    for i in range(pixel_x.shape[0]): # shuffling neurons
        for j in range(pixel_x.shape[1]):
            pixel_x_shuffle[i, j] = np.random.permutation(pixel_x[i, j])
    return pixel_x_shuffle

pixel_x_shuffle = shuffle_neuron(pixel_x)

#from sklearn.decomposition import PCA
# read the neural data
neural_x_all = hkl.load(os.path.join(DATA_DIR, neural_data_path))
neural_x_all['pixel'] = pixel_x # merge pixel_x to neural data
neural_x_all['pixel_shuffle'] = pixel_x_shuffle

# rearange according to the neural speed
speed_ind = np.argsort(speed_list)
mean_dot, err_dot = [], []

# prednet
for mode in ['pixel_shuffle', 'pixel'] + output_mode:
    neural_x = neural_x_all[mode].reshape([neural_x_all[mode].shape[0], neural_x_all[mode].shape[1], -1]) # (n_speed, n_time, features)
    neural_x = neural_x[speed_ind][:, cut0:cut]
    if geo_tool_method == 'cos_xt_xv':
        neural_x = geo_tool.pca_reduce(neural_x, n_components=n_com_cos)
        mean_dot_layer, err_dot_layer = geo_tool.cos_xt_xv(neural_x)
    elif geo_tool_method == 'procrustes_curve_diff_time':
        mean_dot_layer, err_dot_layer = geo_tool.procrustes_curve_diff_time(neural_x, print_message=print_message, n_com=n_com_procrustes)
    elif geo_tool_method == 'dim_manifold':
        mean_dot_layer, err_dot_layer = geo_tool.dim_manifold(neural_x)

    mean_dot.append(mean_dot_layer)
    err_dot.append(err_dot_layer)

print(mean_dot, err_dot)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(range(-2, 4), mean_dot)
ax.errorbar(range(-2, 4), mean_dot, yerr=err_dot)
ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Layer of the Prednet \n -1 means the pixels \n -2 means shuffled pixels')
if geo_tool_method == 'cos_xt_xv':
    ax.set_ylabel('cos of the angle between the tangent \n vector along time and speed')
elif geo_tool_method == 'procrustes_curve_diff_time':
    ax.set_ylabel('disparity')
elif geo_tool_method == 'dim_manifold':
    ax.set_ylabel('Dimensionality when the expalined var is larger than 0.95')

plt.show()
