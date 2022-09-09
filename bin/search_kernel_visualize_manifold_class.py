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
from predusion.mesh_helper import Mesh_Helper
from kitti_settings import *

out_data_head = 'grating_stim'
#out_data_head = 'dot_stim'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
label_path = 'label_' + out_data_head + '_R_prednet' + '.hkl'
label_name_path = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

mesh_size = 10
train_ratio = 0.6
test_ratio = 0.2
explained_var_thre = 0.90
explained_var_thre_pca_all_data = 0.90
# drifting grating configurations
label_id = (1,) # this is id before converting to nonperiodic
mesh_bound = [[0, 0.15], [0, 360], [0, 5]]
var_period = [None, [0, 360], None] # the length is equal to the number of labels (columns of train_label). None means this variable is linear, while given a period interval, von mises function would be used as a kernel
kernel_width = [0.0016, 60, 1]

mesh_hp = Mesh_Helper(var_period)

kernel_width = mesh_hp.kernel_to_nonperiodic(kernel_width)
label_id = mesh_hp.label_id_to_nonperiodic(label_id)
label_mesh = mesh_hp.generate_manifold_label_mesh(mesh_bound, mesh_size)
label_mesh = mesh_hp.label_to_nonperiodic(label_mesh)

### these are for kernel search
kernel_width_speed_list = np.linspace(0.0005, 0.01, 5).reshape(-1, 1)
kernel_width_orix_list = np.linspace(0.01, 1, 5).reshape(-1, 1)
kernel_width_oriy_list = np.linspace(0.01, 1, 5).reshape(-1, 1)

#kernel_width_list = kernel_width_orix_list # [n_kernels, n_variables]
kernel_width_list = np.hstack((kernel_width_orix_list, kernel_width_oriy_list)) # [n_kernels, n_variables]

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
#dataset.convert_label_to_nonperiodic(var_period)
label_origin = dataset.label.copy()
dataset.label = mesh_hp.label_to_nonperiodic(dataset.label)

geoa = manifold.Layer_manifold()

def kernel_size_ploter(dataset, geoa, lt_mesh, label_id, kernel_width_list, train_ratio=0.6, test_ratio=0.2):
    '''
    dataset: a layer_dataset
    geoa: a layer_manifold
    lt_mesh ([n_mesh, n_labels])
    '''
    (feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio, random_seed=42)
    geoa.load_data(feamap_train, label_train)
    score = geoa.search_kernel_width(lt_mesh, feamap_validate, label_validate, label_id, kernel_width_list)
    print('scores are:')
    [print(key,':',value) for key, value in score.items()]

    plt.figure()
    plt.title('r2_score for different kernel size. Size of x and y are same')
    for key, value in score.items():
        plt.plot(kernel_width_list[:, 0], value, label=key)
        plt.scatter(kernel_width_list[:, 0], value)
    plt.legend()
    plt.show()
    return score

kernel_size_ploter(dataset, geoa, label_mesh, label_id, kernel_width_list)

# visualize testing
geoa.load_data(feamap_train, label_train)
# fit the manifold
geoa.fit_info_manifold_grid_all(label_mesh, label_id, kernel_width=kernel_width)
# visualize infomation manifold
n_layer, layer_order = layer_order_helper()
for layer_name in layer_order:
    info_manifold = geoa.ana_group[layer_name][label_id].info_manifold.copy()
    label_mesh = geoa.ana_group[layer_name][label_id].label_mesh.copy()

    plt_dr = Ploter_dim_reduction(method='pca', n_components=2)
    for i, lb_id in enumerate(label_id):
        vmin, vmax = label_mesh[:, i].min(), label_mesh[:, i].max()
        fig, ax =  plt_dr.plot_dimension_reduction(info_manifold, colorinfo=label_mesh[:, i], mode='2D', fit=True, vmin=vmin, vmax=vmax)
        fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, lb_id], mode='2D', fig=fig, ax=ax, marker='+', vmin=vmin, vmax=vmax) # test
        plt.show()

# this is for periodic variable
#label_test = dataset.label_to_origin(label_test, var_period)
label_test = mesh_hp.label_to_origin(label_test)

for layer_name in layer_order:
    info_manifold = geoa.ana_group[layer_name][label_id].info_manifold.copy()

    label_mesh = geoa.ana_group[layer_name][label_id].label_mesh.copy()
    var_period_label_mesh = [var_period[i] for i in label_period_id]
    mesh_hp.set_var_period(var_period_label_mesh)
    label_mesh = mesh_hp.label_to_origin(label_mesh) # convert to original label
    mesh_hp.set_var_period(var_period)

    plt_dr = Ploter_dim_reduction(method='pca', n_components=2)
    for i, lb_id in enumerate(label_period_id):
        vmin, vmax = label_mesh[:, i].min(), label_mesh[:, i].max()
        fig, ax =  plt_dr.plot_dimension_reduction(info_manifold, colorinfo=label_mesh[:, i], mode='2D', fit=True, vmin=vmin, vmax=vmax)
        fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, lb_id], mode='2D', fig=fig, ax=ax, marker='+', vmin=vmin, vmax=vmax) # test
        plt.show()
