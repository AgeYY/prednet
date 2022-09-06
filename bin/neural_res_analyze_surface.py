# analyze the manifold of two variables (surfaces), please first look for proper kernel size in search_kernel_visualize_manifold
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

feamap_path = os.path.join(DATA_DIR, neural_data_path)
label_path = os.path.join(DATA_DIR, label_path)
label_name_path = os.path.join(DATA_DIR, label_name_path)

def layer_order_helper():
    if 'prednet' in neural_data_path:
        n_layer = 5
        layer_order = ['X', 'R0', 'R1', 'R2', 'R3']
    return n_layer, layer_order

dataset = Layer_Dataset(feamap_path, label_path, label_name_path, explained_var_thre=explained_var_thre_pca_all_data, nan_handle='None') # no nan in data, skip nan_handle would be faster

geoa = manifold.Layer_manifold()

############################ Fix the hyperparameter and repeat on different training testing sets on different random seeds
train_ratio, test_ratio = 0.7, 0.3
n_bootstrap = 20
dim, score, angle = {}, {}, {}
for i in range(n_bootstrap):
    (feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio, random_seed=None)
    geoa.load_data(feamap_train, label_train)
    geoa.fit_info_manifold_grid_all(lt_mesh, label_id, kernel_width=kernel_width)
    score_boots = geoa.manifold_decoder_score_all(feamap_test, label_test, label_id)
    dim_boots = geoa.dim_all(explained_var_thre, label_id)
    #angle_boots = geoa.angle_tangent_vec_all(label_id, label_id)

    for key in dim_boots:
        try:
            dim[key].append(dim_boots[key])
            score[key].append(score_boots[key])
            #angle[key] = np.append(angle[key], angle_boots[key])
        except KeyError:
            dim[key] = [dim_boots[key]]
            score[key] = [score_boots[key]]
            #angle[key] = angle_boots[key]

### visualize the dimensionality of the manifold
n_layer, layer_order = layer_order_helper()
fig, ax = plt.subplots(figsize=(4, 4))
ax = ploter.plot_layer_error_bar_helper(score, n_layer, layer_order, ax)
ax.axhline(1, color='k', linestyle='--')
ax.set_ylim(bottom=None, top=1)
ax.set_title('score')
plt.show()
fig, ax = plt.subplots(figsize=(4, 4))
ax = ploter.plot_layer_error_bar_helper(dim, n_layer, layer_order, ax)
ax.axhline(0, color='k', linestyle='--')
ax.set_title('dim')
plt.show()
fig, ax = plt.subplots(figsize=(4, 4))
ax = ploter.plot_layer_error_bar_helper(angle, n_layer, layer_order, ax)
ax.axhline(0, color='k', linestyle='--')
ax.set_title('angle')
plt.show()
