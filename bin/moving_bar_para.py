# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py
# we would like to show cos(u_v(t1, v), u_v(t2, v)) = 1
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from predusion.agent import Agent
from predusion.frechetdist import frdist
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

def cos_para_layer(neural_x, speed_list, cut0, cut, error_bar='std'):
    '''
    calculate the mean and se (or std) of one layer
    input:
      neural_x ([n_speed, n_time, feature])

    '''
    # rearange the neural speed
    ind = np.argsort(speed_list)
    neural_x = neural_x[ind][:, cut0:cut]

    # calculate the different along the temporal direction
    #neural_x_speed = np.gradient(neural_x, axis=0)
    neural_x_speed_mean = np.mean(neural_x, axis=0)
    #shift_neural_x = np.tile(np.expand_dims(neural_x[0], axis=0), (cut, 1, 1))
    shift_neural_x = np.tile(np.expand_dims(neural_x_speed_mean, axis=0), (cut, 1 , 1))
    #neural_x_speed = neural_x - np.tile(neural_x[:, 0])
    neural_x_speed = neural_x - shift_neural_x
    #neural_x_speed = neural_x_speed[1:, :]
    nt = neural_x_speed.shape[1]
    # calculate the cos
    dot = []

    for i in range(nt - 1):
        for j in range(i + 1, nt):
            dot.append( np.sum(neural_x_speed[:, i] * neural_x_speed[:, j], axis=-1) / np.linalg.norm(neural_x_speed[:, i], axis=-1) / np.linalg.norm(neural_x_speed[:, j], axis=-1) )
            #dot.append(frdist(neural_x_speed[:, i], neural_x_speed[:, j]))

    dot_flat = np.array(dot).flatten()

    if error_bar=='std':
        err = np.std(dot_flat)
    else: # sem
        err = np.std(dot_flat) / np.sqrt(np.size(dot_flat))

    return np.mean(dot_flat), err




nt = 12 # each prediction group contains nt images

out_data_head = 'moving_bar'
output_mode = ['E0', 'E1', 'E2', 'E3']
neural_data_path = 'neural_moving_bar_E' + '.hkl'
#output_mode = ['R0', 'R1', 'R2', 'R3']
#neural_data_path = 'neural_moving_bar_R' + '.hkl'
cut = 12
cut0 = 2 # frames from cut_0 to cut
n_com = None # used for PCA

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

#from sklearn.decomposition import PCA
# read the neural data
neural_x_all = hkl.load(os.path.join(DATA_DIR, neural_data_path))

mean_dot, err_dot = [], []
# pixel
mean_dot_pixel, err_dot_pixel = cos_para_layer(pixel_x, speed_list, cut0, cut)
mean_dot.append(mean_dot_pixel)
err_dot.append(err_dot_pixel)

# prednet
for mode in output_mode:
    neural_x = neural_x_all[mode].reshape([neural_x_all[mode].shape[0], neural_x_all[mode].shape[1], -1]) # (n_speed, n_time, features)
    mean_dot_layer, err_dot_layer = cos_para_layer(neural_x, speed_list, cut0, cut)
    mean_dot.append(mean_dot_layer)
    err_dot.append(err_dot_layer)

print(mean_dot, err_dot)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(range(-1, 4), mean_dot)
ax.errorbar(range(-1, 4), mean_dot, yerr=err_dot)
ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Layer of the Prednet \n -1 means the pixels')
ax.set_ylabel('cos of the angle between the tangent \n vector along time and speed')
plt.show()