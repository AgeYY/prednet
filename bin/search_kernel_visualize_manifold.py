# search the optimal kernel size and visualize the manifold along with testing datasets
# analyze the manifold of two variables (surfaces)
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import predusion.manifold as manifold
import predusion.ploter as ploter
from predusion.ploter import Ploter_dim_reduction
from predusion.static_dataset import Layer_Dataset, train_test_validate_split
from kitti_settings import *

out_data_head = 'grating_stim'
#out_data_head = 'dot_stim'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
label_path = 'label_' + out_data_head + '_R_prednet' + '.hkl'
label_name_path = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

mesh_size = 50
label_id = (0,) # only fit manifold about these information variables.
train_ratio = 0.6
test_ratio = 0.2
explained_var_thre = 0.90
explained_var_thre_pca_all_data = 0.90
# drifting grating configurations
lt0_mesh = np.linspace(0, 0.15, mesh_size)
lt1_mesh = np.linspace(0, 360, mesh_size)
lt2_mesh = np.linspace(0, 5, 30)
lt_mesh = [lt0_mesh, lt1_mesh, lt2_mesh]
kernel_width = [0.0016, 10, 1]

### these are for kernel search
kernel_width_speed_list = np.linspace(0.0005, 0.01, 5).reshape(-1, 1)
kernel_width_ori_list = np.linspace(5, 30, 10).reshape(-1, 1)

kernel_width_list = kernel_width_speed_list # [n_kernels, n_variables]

feamap_path = os.path.join(DATA_DIR, neural_data_path)
label_path = os.path.join(DATA_DIR, label_path)
label_name_path = os.path.join(DATA_DIR, label_name_path)

### start program
def layer_order_helper():
    if 'prednet' in neural_data_path:
        n_layer = 5
        layer_order = ['X', 'R0', 'R1', 'R2', 'R3']
    return n_layer, layer_order

dataset = Layer_Dataset(feamap_path, label_path, label_name_path, explained_var_thre=explained_var_thre_pca_all_data, nan_handle='None') # no nan in data, skip nan_handle would be faster

geoa = manifold.Layer_manifold()

############################### Use grid search in a single train/validate/test split to find optimal kernel_width
(feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio, random_seed=42)
score = geoa.search_kernel_width(lt_mesh, feamap_train, label_train, feamap_validate, label_validate, label_id, kernel_width_list)
[print(key,':',value) for key, value in score.items()]

for key, value in score.items():
    plt.plot(kernel_width_list, value, label=key)
    plt.scatter(kernel_width_list, value)
plt.legend()
plt.show()

# visualize testing
geoa.load_data(feamap_train, label_train)
# fit the manifold
geoa.fit_info_manifold_grid_all(lt_mesh, label_id, kernel_width=kernel_width)
# visualize infomation manifold
n_layer, layer_order = layer_order_helper()
for layer_name in layer_order:
    info_manifold = geoa.ana_group[layer_name][label_id].info_manifold.copy()
    label_mesh = geoa.ana_group[layer_name][label_id].label_mesh.copy()

    plt_dr = Ploter_dim_reduction(method='pca', n_components=2)
    for i, lb_id in enumerate(label_id):
        vmin, vmax = label_mesh[:, i].min(), label_mesh[:, i].max()
        fig, ax =  plt_dr.plot_dimension_reduction(info_manifold, colorinfo=label_mesh[:, i], mode='2D', fit=True, vmin=vmin, vmax=vmax)
        #fig, ax =  plt_dr.plot_dimension_reduction(feamap_validate[layer_name], colorinfo=label_validate[:, lb_id], mode='2D', fig=fig, ax=ax, marker='+', vmin=vmin, vmax=vmax) # test
        fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, lb_id], mode='2D', fig=fig, ax=ax, marker='+', vmin=vmin, vmax=vmax) # test
        plt.show()
