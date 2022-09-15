# search the optimal kernel size and visualize the manifold along with testing datasets
# analyze the manifold of two variables (surfaces)
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import predusion.manifold as manifold
import predusion.ploter as ploter
from predusion.ploter import kernel_size_ploter, layer_manifold_ploter
from predusion.static_dataset import Layer_Dataset, train_test_validate_split
from predusion.mesh_helper import Mesh_Helper
from kitti_settings import *
from search_kernel_visualize_manifold_parameters import *

mesh_hp = Mesh_Helper(var_period)

kernel_width = mesh_hp.kernel_to_nonperiodic(kernel_width)

label_id = mesh_hp.label_id_to_nonperiodic(label_id)

label_mesh = mesh_hp.generate_manifold_label_mesh(mesh_bound, mesh_size, random=mesh_random, grid=mesh_grid)
label_mesh = mesh_hp.label_to_nonperiodic(label_mesh)

kernel_mesh = mesh_hp.generate_kernel_mesh(kernel_mesh_bound, kernel_mesh_size, random=kernel_random)
kernel_mesh = mesh_hp.kernel_mesh_to_nonperiodic(kernel_mesh)

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

kernel_size_ploter(geoa, mesh_hp, feamap_train, label_train, feamap_validate, label_validate, label_mesh, label_id, kernel_mesh)

## visualize testing
#geoa.load_data(feamap_train, label_train)
#label_mesh = mesh_hp.generate_manifold_label_mesh(mesh_bound, mesh_size, grid=True)
#label_mesh = mesh_hp.label_to_nonperiodic(label_mesh)
#geoa.build_kernel(label_mesh, label_id, kernel_width=kernel_width)
#geoa.fit_info_manifold_all(label_id)
#
### visualize infomation manifold
#label_test = mesh_hp.label_to_origin(label_test)
#label_mesh_origin = mesh_hp.label_to_origin(label_mesh)
#label_id_origin = mesh_hp.label_id_to_origin(label_id)
#
#fig, ax = layer_manifold_ploter(geoa, n_layer, layer_order, feamap_test, label_test, label_mesh_origin, label_id_origin, label_id)
plt.show()
