# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

import predusion.geo_tool as geo_tool
import predusion.ploter as ploter
from predusion.ploter import Ploter_dim_reduction
from predusion.static_dataset import Layer_Dataset, train_test_validate_split, Meta_Dataset
#from predusion.manifold_analyzer import Manifold_analyzer
from kitti_settings import *
import argparse
from sklearn.linear_model import Ridge as scikit_ridge
from sklearn.model_selection import LeavePOut

parser = argparse.ArgumentParser()
parser.add_argument('--data_head', default='moving_bar20', type=str,
                    help='head of the dataset')
parser.add_argument('--nt', default=12, type=int, help='number of frames per video')
parser.add_argument('--cut_time', nargs=2, default=None, type=int, help='analyze the geometric property only limit with this time interval. unit is frame')
parser.add_argument('--cut_speed', nargs=2, default=None, type=int, help='analyze the geometric property only limit with this speed interval. unit is the rank of speeds, from lowest to the highest')
parser.add_argument('--n_com_procrustes', default=3, type=int, help='number of dimensions for procrustes (comparing the shape similarity across different time points.)')

arg = parser.parse_args()

out_data_head = arg.data_head
cut_time = arg.cut_time
n_com_procrustes = arg.n_com_procrustes
nt = arg.nt

#out_data_head = 'grating_stim'
#out_data_head = 'moving_rect2080'
#dataset_name_list = ['moving_rect2080', 'moving_bar_wustl', 'moving_bar20']
dataset_name_list = ['moving_rect2080', 'moving_bar_wustl', 'moving_bar20', 'moving_bar_red']
feamap_path_list, label_path_list, label_name_path_list = [], [], []
for out_data_head in dataset_name_list:
    neural_data_path = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
    label_path = 'label_' + out_data_head + '_R_prednet' + '.hkl'
    label_name_path = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

    feamap_path = os.path.join(DATA_DIR, neural_data_path)
    label_path = os.path.join(DATA_DIR, label_path)
    label_name_path = os.path.join(DATA_DIR, label_name_path)

    feamap_path_list.append(feamap_path)
    label_path_list.append(label_path)
    label_name_path_list.append(label_name_path)

label_id = 0
train_ratio = 0.6
test_ratio = 0.2
explained_var_thre = 0.95
## drifting grating configurations
#lt_mesh = np.linspace(0, 0.12, 100) 
#kernel_width = 0.0001 
## moving_rect2080, moving_bar_wustl, moving_bar20 configurations
#lt_mesh = np.linspace(0, 12, 100) # moving bar 2080
#kernel_width = 0.5
## moving_bar_wustl configurations
lt_mesh = np.linspace(0, 12, 100)
kernel_width = 0.5

def layer_order_helper():
    if 'prednet' in neural_data_path:
        n_layer = 5
        layer_order = ['X', 'R0', 'R1', 'R2', 'R3']
    return n_layer, layer_order

dataset = Meta_Dataset(dataset_name_list, feamap_path_list, label_path_list, label_name_path_list)

geoa = geo_tool.Geo_analyzer()

############################## Fix the hyperparameter, leave-p-out cross validation
train_ratio, test_ratio = 0.7, 0.3
label_id=0
n_bootstrap = 1
dim, score = {}, {}

dataset_idx = np.arange(len(dataset_name_list))

train_idx, test_idx = [1, 2], [0, 1]
(feamap_train, label_train), (feamap_test, label_test) = dataset[train_idx], dataset[test_idx]
geoa.load_data(feamap_train, label_train)
geoa.fit_info_manifold_all(lt_mesh, label_id, kernel_width=kernel_width)

plt_dr = Ploter_dim_reduction(method='pca')
fig, ax =  plt_dr.plot_dimension_reduction(feamap_train['R3'], colorinfo=label_train[:, label_id], fit=True, mode='3D')
#fig, ax =  plt_dr.plot_dimension_reduction(feamap_validate[layer_name], colorinfo=label_validate[:, label_id], mode='2D', fig=fig, ax=ax) # validation
#fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, label_id], mode='2D', fig=fig, ax=ax) # test
plt.show()

'''
lpo = LeavePOut(1)

for train_idx, test_idx in lpo.split(dataset_idx):
    (feamap_train, label_train), (feamap_test, label_test) = dataset[train_idx], dataset[test_idx]
    geoa.load_data(feamap_train, label_train)
    geoa.fit_info_manifold_all(lt_mesh, label_id, kernel_width=kernel_width)

    #### visualization
    #n_layer, layer_order = layer_order_helper()
    #for layer_name in layer_order:
    #    info_manifold = geoa.ana_group[layer_name][label_id].info_manifold.copy()
    #    plt_dr = Ploter_dim_reduction(method='pca')
    #    fig, ax =  plt_dr.plot_dimension_reduction(info_manifold, colorinfo=lt_mesh, fit=True, mode='2D')
    #    #fig, ax =  plt_dr.plot_dimension_reduction(feamap_validate[layer_name], colorinfo=label_validate[:, label_id], mode='2D', fig=fig, ax=ax) # validation
    #    fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, label_id], mode='2D', fig=fig, ax=ax) # test
    #    plt.show()


    dim_boots, score_boots = geoa.linear_regression_score_all(explained_var_thre, feamap_test, label_test, label_id) # measuring the amount of information in the subspace of the manifold
    for key in dim_boots:
        try:
            dim[key].append(dim_boots[key])
            score[key].append(score_boots[key])
        except KeyError:
            dim[key] = [dim_boots[key]]
            score[key] = [score_boots[key]]

### visualize the dimensionality of the manifold
n_layer, layer_order = layer_order_helper()
fig, ax = plt.subplots(figsize=(4, 4))
ax = ploter.plot_layer_error_bar_helper(score, n_layer, layer_order, ax)
ax.axhline(1, color='k', linestyle='--')
#ax.set_ylim(0, 1)
plt.show()
fig, ax = plt.subplots(figsize=(4, 4))
ax = ploter.plot_layer_error_bar_helper(dim, n_layer, layer_order, ax)
ax.axhline(0, color='k', linestyle='--')
plt.show()
'''
