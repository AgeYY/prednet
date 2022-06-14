# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py

import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from data_utils import SequenceGenerator
from predusion.manifold_analyzer import Manifold_analyzer
from predusion.ploter import plot_dimension_reduction

from kitti_settings import *

#from sklearn.manifold import MDS
#from sklearn.manifold import LocallyLinearEmbedding
#from sklearn.manifold import Isomap
#from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_head', default='moving_bar', type=str, help='head of the dataset')
parser.add_argument('--nt', default=12, type=int, help='number of frames per video')
parser.add_argument('--embed_method', default='pca', type=str, help='dimension reduction method')
parser.add_argument('--cut_time', nargs=2, default=None, type=int, help='analyze the geometric property only limit with this time interval. unit is frame')

arg = parser.parse_args()
out_data_head = arg.data_head
embed_method = arg.embed_method
cut_time = arg.cut_time
nt = arg.nt

batch_size = None

#output_mode = ['E0', 'E1', 'E2', 'E3']
#neural_data_path = 'neural_' + out_data_head + '_E' + '.hkl'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R' + '.hkl'
n_components = 3
align_delta = None

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
neural_dir = os.path.join(DATA_DIR, neural_data_path)

# load neural response
mani_analyzer = Manifold_analyzer()
mani_analyzer.load_data(train_file, train_sources, label_file, neural_dir, add_shuffle_pixel=False, nt=nt)

speed_list = np.array(mani_analyzer.speed_list)
neural_x_all = mani_analyzer.neural_x_all
output_mode = mani_analyzer.output_mode

# create color information map
colorinfo_time = np.arange(nt) # temperal color scheme
colorinfo_speed = np.array(speed_list)

colorinfo_time, colorinfo_speed = np.meshgrid(colorinfo_time, colorinfo_speed)

# cut and sort the color information
ind = np.argsort(speed_list)

colorinfo_time_cut = colorinfo_time[:, cut_time[0]:cut_time[1]]
colorinfo_speed_cut = colorinfo_speed[:, cut_time[0]:cut_time[1]]

colorinfo_time_sort = colorinfo_time_cut # the time information is sorted
colorinfo_speed_sort = colorinfo_speed_cut[ind]

for mode in output_mode:
    neural_x = neural_x_all[mode]
    neural_x_cut = neural_x[:, cut_time[0]:cut_time[1]].reshape([neural_x.shape[0], cut_time[1] - cut_time[0], -1]) # (n_speed, n_time, features)

    # rearange the neural speed
    neural_x_sort = neural_x_cut[ind]

    plot_dimension_reduction(neural_x_sort, method=embed_method, n_components=n_components, colorinfo=colorinfo_time_sort, title=out_data_head + '_' + embed_method + '_neuron_color_time_{}'.format(mode), align_delta=align_delta)
    plot_dimension_reduction(neural_x_sort, method=embed_method, n_components=n_components, colorinfo=colorinfo_speed_sort, title=out_data_head + '_' + embed_method + '_neuron_color_speed_{}'.format(mode), align_delta=align_delta)
