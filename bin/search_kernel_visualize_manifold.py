# search the optimal kernel size and visualize the manifold along with testing datasets
# analyze the manifold of two variables (surfaces)
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import predusion.manifold as manifold
import predusion.ploter as ploter
from predusion.ploter import Ploter_dim_reduction, kernel_size_ploter
from predusion.static_dataset import Layer_Dataset, train_test_validate_split
from predusion.mesh_helper import Mesh_Helper
from kitti_settings import *

out_data_head = 'grating_stim'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
label_path = 'label_' + out_data_head + '_R_prednet' + '.hkl'
label_name_path = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

mesh_size = 50
kernel_mesh_size = 5
train_ratio = 0.6
test_ratio = 0.2
explained_var_thre = 0.90
explained_var_thre_pca_all_data = 0.90
# drifting grating configurations
label_id = (0, 1)
mesh_bound = [[0, 0.15], [0, 360], [0, 5]]
kernel_mesh_bound = [[0.0005, 0.01], [10, 60], [0.01, 1]]
var_period = [None, [0, 360], None] # the length is equal to the number of labels (columns of train_label). None means this variable is linear, while given a period interval, von mises function would be used as a kernel
kernel_width = [0.0016, 30, 1]

mesh_hp = Mesh_Helper(var_period)

kernel_width = mesh_hp.kernel_to_nonperiodic(kernel_width)
label_id = mesh_hp.label_id_to_nonperiodic(label_id)
label_mesh = mesh_hp.generate_manifold_label_mesh(mesh_bound, mesh_size)
label_mesh = mesh_hp.label_to_nonperiodic(label_mesh)
kernel_mesh = mesh_hp.generate_kernel_mesh(kernel_mesh_bound, kernel_mesh_size)
kernel_mesh = mesh_hp.kernel_mesh_to_nonperiodic(kernel_mesh)

def layer_order_helper():
    if 'prednet' in neural_data_path:
        n_layer = 5
        layer_order = ['X', 'R0', 'R1', 'R2', 'R3']
    return n_layer, layer_order

feamap_path = os.path.join(DATA_DIR, neural_data_path)
label_path = os.path.join(DATA_DIR, label_path)
label_name_path = os.path.join(DATA_DIR, label_name_path)

### start program
dataset = Layer_Dataset(feamap_path, label_path, label_name_path, explained_var_thre=explained_var_thre_pca_all_data, nan_handle='None') # no nan in data, skip nan_handle would be faster
#dataset.convert_label_to_nonperiodic(var_period)
label_origin = dataset.label.copy()
dataset.label = mesh_hp.label_to_nonperiodic(dataset.label)

geoa = manifold.Layer_manifold()

(feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio, random_seed=42)

kernel_size_ploter(feamap_train, label_train, feamap_validate, label_validate, geoa, label_mesh, label_id, kernel_mesh)

# visualize testing
geoa.load_data(feamap_train, label_train)
label_mesh = mesh_hp.generate_manifold_label_mesh(mesh_bound, mesh_size, grid=True)
label_mesh = mesh_hp.label_to_nonperiodic(label_mesh)
geoa.build_kernel(label_mesh, label_id, kernel_width=kernel_width)
geoa.fit_info_manifold_all(label_id)
## visualize infomation manifold
n_layer, layer_order = layer_order_helper()

label_test = mesh_hp.label_to_origin(label_test)
label_mesh_origin = mesh_hp.label_to_origin(label_mesh)
label_id_origin = mesh_hp.label_id_to_origin(label_id)

for layer_name in layer_order:
    info_manifold = geoa.ana_group[layer_name][label_id].info_manifold.copy()

    plt_dr = Ploter_dim_reduction(method='pca', n_components=2)
    for lb_id in label_id_origin:
        vmin, vmax = label_mesh_origin[:, lb_id].min(), label_mesh_origin[:, lb_id].max()
        fig, ax =  plt_dr.plot_dimension_reduction(info_manifold, colorinfo=label_mesh_origin[:, lb_id], mode='2D', fit=True, vmin=vmin, vmax=vmax)
        fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, lb_id], mode='2D', fig=fig, ax=ax, marker='+', vmin=vmin, vmax=vmax) # test
        plt.show()
