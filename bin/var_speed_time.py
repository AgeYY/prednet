# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py
# we would like to show the variance of speed is smaller than that of time
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from predusion.agent import Agent
from predusion.frechetdist import frdist
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
from sklearn.decomposition import PCA

def cos_para_layer(neural_x, speed_list, cut0, cut, error_bar='std', n_components=20):
    '''
    calculate the mean and se (or std) of one layer
    input:
      neural_x ([n_speed, n_time, feature])

    '''
    # rearange the neural speed
    ind = np.argsort(speed_list)
    neural_x = neural_x[ind][:, cut0:cut]

    # pca processing
    neural_x_flat = neural_x.reshape(-1, neural_x.shape[-1])
    pca = PCA(n_components=n_components)
    neural_x_flat = pca.fit_transform(neural_x_flat)
    print(np.cumsum(pca.explained_variance_ratio_))
    neural_x = neural_x_flat.reshape(neural_x.shape[0], neural_x.shape[1], n_components)

    var_time, var_speed = [], []
    # variance of time
    for i_sp in range(neural_x.shape[0]):
        neural_x_time = neural_x[i_sp]
        var_time_temp = np.sum(np.var(neural_x_time, axis=0))
        var_time.append(var_time_temp)
    for i_t in range(neural_x.shape[1]):
        neural_x_speed = neural_x[:, i_t]
        var_speed_temp = np.sum(np.var(neural_x_speed, axis=0))
        var_speed.append(var_speed_temp)

    return np.mean(var_time), np.var(var_time), np.mean(var_speed), np.var(var_speed)



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

mean_var_time, err_var_time = [], []
mean_var_speed, err_var_speed = [], []
# pixel
mean_var_time_pixel, err_var_time_pixel, mean_var_speed_pixel, err_var_speed_pixel = cos_para_layer(pixel_x, speed_list, cut0, cut)

mean_var_time.append(mean_var_time_pixel)
err_var_time.append(err_var_time_pixel)
mean_var_speed.append(mean_var_speed_pixel)
err_var_speed.append(err_var_speed_pixel)

# prednet
for mode in output_mode:
    neural_x = neural_x_all[mode].reshape([neural_x_all[mode].shape[0], neural_x_all[mode].shape[1], -1]) # (n_speed, n_time, features)
    mean_var_time_layer, err_var_time_layer, mean_var_speed_layer, err_var_speed_layer = cos_para_layer(neural_x, speed_list, cut0, cut)

    mean_var_time.append(mean_var_time_layer)
    err_var_time.append(err_var_time_layer)
    mean_var_speed.append(mean_var_speed_layer)
    err_var_speed.append(err_var_speed_layer)

print(mean_var_time, err_var_time)
print(mean_var_speed, err_var_speed)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(range(-1, 4), mean_var_time, label='variance along time')
ax.errorbar(range(-1, 4), mean_var_time, yerr=err_var_time)
ax.scatter(range(-1, 4), mean_var_speed, label='variance along speed')
ax.errorbar(range(-1, 4), mean_var_speed, yerr=err_var_speed)

ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Layer of the Prednet \n -1 means the pixels')
ax.set_ylabel('variance')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(range(-1, 4), np.array(mean_var_speed)/np.array(mean_var_time))
ax.plot(range(-1, 4), np.array(mean_var_speed)/np.array(mean_var_time))

ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Layer of the Prednet \n -1 means the pixels')
ax.set_ylabel('var(speed) / var(time)')
plt.show()
