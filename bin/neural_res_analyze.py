# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

import predusion.geo_tool as geo_tool
import predusion.ploter as ploter
from predusion.ploter import Ploter_dim_reduction
from predusion.static_dataset import Layer_Dataset, train_test_validate_split
#from predusion.manifold_analyzer import Manifold_analyzer
from kitti_settings import *
import argparse
from sklearn.linear_model import Ridge as scikit_ridge

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

out_data_head = 'grating_stim'
#out_data_head = 'moving_rect2080'
#out_data_head = 'moving_bar_wustl'
#out_data_head = 'dot_stim'
#output_mode = ['E0', 'E1', 'E2', 'E3']
#neural_data_path = 'neural_' + out_data_head + '_E' + '.hkl'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
label_path = 'label_' + out_data_head + '_R_prednet' + '.hkl'
label_name_path = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

label_id = 0
train_ratio = 0.6
test_ratio = 0.2
explained_var_thre = 0.95
explained_var_thre_pca_all_data = 0.90
# drifting grating configurations
lt0_mesh = np.linspace(0, 0.12, 100) 
lt1_mesh = np.linspace(0, 180, 100)
lt_mesh = [lt0_mesh, lt1_mesh]
kernel_width = [0.0001, 90]
## moving_rect2080 configurations
#lt_mesh = np.linspace(0, 12, 100) # moving bar 2080
#kernel_width = 0.5
## dot_stim configurations
#lt_mesh = np.linspace(0, 8, 100)
#kernel_width = 0.5
#lt_mesh = np.linspace(2, 12, 100) # for time latent
#kernel_width = 1

feamap_path = os.path.join(DATA_DIR, neural_data_path)
label_path = os.path.join(DATA_DIR, label_path)
label_name_path = os.path.join(DATA_DIR, label_name_path)

def layer_order_helper():
    if 'prednet' in neural_data_path:
        n_layer = 5
        layer_order = ['X', 'R0', 'R1', 'R2', 'R3']
    return n_layer, layer_order

dataset = Layer_Dataset(feamap_path, label_path, label_name_path, explained_var_thre=explained_var_thre_pca_all_data)

geoa = geo_tool.Geo_analyzer()

############################## Tune the kernel_width
(feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio)
geoa.load_data(feamap_train, label_train)
geoa.label_dis([label_id]) # show the distribution of labels

# fit the manifold
geoa.fit_info_manifold_all(lt_mesh[label_id], label_id, kernel_width=kernel_width[label_id])
# visualize infomation manifold
n_layer, layer_order = layer_order_helper()
for layer_name in layer_order:
    info_manifold = geoa.ana_group[layer_name][label_id].info_manifold.copy()
    plt_dr = Ploter_dim_reduction(method='pca', n_components=2)
    fig, ax =  plt_dr.plot_dimension_reduction(feamap_validate[layer_name], colorinfo=label_validate[:, label_id], mode='2D', alpha=0.5, fit=True) # validation
    fig, ax =  plt_dr.plot_dimension_reduction(info_manifold, colorinfo=lt_mesh[label_id], mode='2D', fig=fig, ax=ax)
    #fig, ax =  plt_dr.plot_dimension_reduction(feamap_test[layer_name], colorinfo=label_test[:, label_id], mode='2D', fig=fig, ax=ax) # test
    plt.show()
############################## Tune the kernel_width

############################### Fix the hyperparameter and repeat on different training testing sets on different random seeds
#train_ratio, test_ratio = 0.6, 0.4
#n_bootstrap = 10
#dim, score = {}, {}
#for i in range(n_bootstrap):
#    (feamap_train, label_train), (feamap_test, label_test), (feamap_validate, label_validate) = train_test_validate_split(dataset, train_ratio, test_ratio)
#    geoa.load_data(feamap_train, label_train)
#    geoa.fit_info_manifold_all(lt_mesh[label_id], label_id, kernel_width=kernel_width[label_id])
#    #score_boots = geoa.linear_regression_score_all(explained_var_thre, feamap_test, label_test, label_id) # measuring the amount of information in the subspace of the manifold
#    score_boots = geoa.manifold_decoder_score_all(feamap_test, label_test, label_id)
#    #score_boots = geoa.mutual_info_all(feamap_test, label_test, label_id, sigma=kernel_width[label_id])
#    dim_boots = geoa.dim_all(explained_var_thre, label_id)
#
#    for key in dim_boots:
#        try:
#            dim[key].append(dim_boots[key])
#            score[key].append(score_boots[key])
#        except KeyError:
#            dim[key] = [dim_boots[key]]
#            score[key] = [score_boots[key]]
#
#### visualize the dimensionality of the manifold
#n_layer, layer_order = layer_order_helper()
#fig, ax = plt.subplots(figsize=(4, 4))
#ax = ploter.plot_layer_error_bar_helper(score, n_layer, layer_order, ax)
#ax.axhline(1, color='k', linestyle='--')
#ax.set_ylim(0, 1)
#plt.show()
#fig, ax = plt.subplots(figsize=(4, 4))
#ax = ploter.plot_layer_error_bar_helper(dim, n_layer, layer_order, ax)
#ax.axhline(0, color='k', linestyle='--')
#plt.show()
