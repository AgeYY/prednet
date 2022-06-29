# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py
import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from predusion.agent import Agent
import predusion.geo_tool as geo_tool
from predusion.manifold_analyzer import Manifold_analyzer
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_head', default='moving_bar', type=str,
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

#output_mode = ['E0', 'E1', 'E2', 'E3']
#neural_data_path = 'neural_' + out_data_head + '_E' + '.hkl'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R' + '.hkl'
#geo_tool_method_list = ['cos_xt_xv', 'dim_manifold', 'ratio_speed_time', 'procrustes_curve_diff_time']
geo_tool_method_list = ['angle_PC', 'r2_score']
n_com_cos = None

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
print_message = False

neural_dir = os.path.join(DATA_DIR, neural_data_path)

mani_analyzer = Manifold_analyzer()
mani_analyzer.load_data(train_file, train_sources, label_file, neural_dir, add_shuffle_pixel=True, nt=nt)

mean_dot, err_dot = {}, {}
for geo_tool_method in geo_tool_method_list:
    if geo_tool_method == 'procrustes_curve_diff_time': n_com = n_com_procrustes
    else: n_com = n_com_cos

    mean_dot[geo_tool_method], err_dot[geo_tool_method] = mani_analyzer.analyze(geo_tool_method=geo_tool_method, n_com=n_com, cut0=cut_time[0], cut=cut_time[1])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, sharex=True)
y_label = {}
y_label['cos_xt_xv'] = 'cos of the angle between \n the tangent vector along \n time and speed'
y_label['procrustes_curve_diff_time'] = 'dissimilarity'
y_label['dim_manifold'] = 'number of principal components \n when the expalined var \n is larger than 0.95'
y_label['ratio_speed_time'] = 'var of speed / var of time'
y_label['angle_PC'] = 'angle between two information pls (deg)'
y_label['r2_score'] = 'r2 score of the speed'

for i, method in enumerate(geo_tool_method_list):
    idx, idy = i//2, i%2
    ax[idx, idy].scatter(range(-2, len(output_mode)), mean_dot[method])
    ax[idx, idy].errorbar(range(-2, len(output_mode)), mean_dot[method], yerr=err_dot[method])
    ax[idx, idy].axhline(0, color='k', linestyle='--')
    ax[idx, idy].set_ylabel(y_label[method])

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Layer of the Prednet \n -1 means the pixels; -2 means shuffled pixels')
plt.tight_layout()
plt.savefig('./figs/' + out_data_head + '.pdf')
plt.show()
