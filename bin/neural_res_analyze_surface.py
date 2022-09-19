# analyze the manifold of two variables (surfaces), please first look for proper kernel size in search_kernel_visualize_manifold
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import predusion.manifold as manifold
from predusion.mesh_helper import Mesh_Helper
import predusion.ploter as ploter
from predusion.ploter import Ploter_dim_reduction
from predusion.static_dataset import Layer_Dataset, train_test_validate_split
from kitti_settings import *
from search_kernel_visualize_manifold_parameters import *

mesh_random = True
mesh_grid = False

mesh_hp = Mesh_Helper(var_period)

kernel_width = mesh_hp.kernel_to_nonperiodic(kernel_width)

label_id = mesh_hp.label_id_to_nonperiodic(label_id)

label_mesh = mesh_hp.generate_manifold_label_mesh(mesh_bound, mesh_size, random=mesh_random, grid=mesh_grid)
label_mesh = mesh_hp.label_to_nonperiodic(label_mesh)

feamap_path = os.path.join(DATA_DIR, neural_data_path)
label_path = os.path.join(DATA_DIR, label_path)
label_name_path = os.path.join(DATA_DIR, label_name_path)

### start program
dataset = Layer_Dataset(feamap_path, label_path, label_name_path, explained_var_thre=explained_var_thre_pca_all_data, nan_handle='None') # no nan in data, skip nan_handle would be faster
label_origin = dataset.label.copy()
dataset.label = mesh_hp.label_to_nonperiodic(dataset.label)

geoa = manifold.Layer_manifold()

############################ Fix the hyperparameter and repeat on different training testing sets on different random seeds
train_ratio, test_ratio = 0.7, 0.3
n_bootstrap = 30
dim, score, angle = {}, {}, {}
for i in range(n_bootstrap):
    (feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio, random_seed=None)
    geoa.load_data(feamap_train, label_train)
    geoa.build_kernel(label_mesh, label_id, kernel_width=kernel_width)
    geoa.fit_info_manifold_all(label_id)

    score_boots = geoa.manifold_decoder_score_all(feamap_test, label_test, label_id)
    dim_boots = geoa.dim_all(explained_var_thre, label_id)

    # calculate tangent vectors
    # convert tangent vector to periodic variable
    # calculate the angle

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
