# read in the data, this class can output several properties, including:
# - adding shuffled pixel space
# - cos of the angle between tangent vector along speed and time
# - using procrustes to compare different speed curves
# - dimensionality of speed manifold
# - ratio between the variation of speed curve and that of time curve
import numpy as np
import hickle as hkl

import predusion.geo_tool as geo_tool
from data_utils import SequenceGenerator
from kitti_settings import *

def shuffle_neuron(pixel_x):
    pixel_x_shuffle = np.empty(pixel_x.shape)
    for i in range(pixel_x.shape[0]): # shuffling neurons
        for j in range(pixel_x.shape[1]):
            pixel_x_shuffle[i, j] = np.random.permutation(pixel_x[i, j])
    return pixel_x_shuffle

class Manifold_analyzer():
    def __init__(self):
        pass

    def load_data(self, train_file, train_sources, label_file, neural_data_path, add_shuffle_pixel=False, nt=12):
        '''
        neural_dir: the neural response to the train_file, generated from gen_neural_res_moving_bar.py
        '''
        train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)
        X_train, label = train_generator.create_all(out_label=True)
        self.speed_list = label
        pixel_x = X_train.reshape([X_train.shape[0], X_train.shape[1], -1])
        self.neural_x_all = hkl.load(neural_data_path)

        self.output_mode = list(self.neural_x_all.keys())

        self.neural_x_all['pixel'] = pixel_x # merge pixel_x to neural data

        self.output_mode.insert(0, 'pixel')

        if add_shuffle_pixel:
            pixel_x_shuffle = shuffle_neuron(pixel_x)
            self.neural_x_all['pixel_shuffle'] = pixel_x_shuffle
            self.output_mode.insert(0, 'pixel_shuffle')

    def analyze(self, geo_tool_method='cos_xt_xv', cut0=2, cut=12, n_com=20, print_pca_message=False):
        '''
        geo_tool_method (str): cos_xt_xv, procrustes_curve_diff_time, dim_manifold, ratio_speed_time
        cut0, cut (int): only consider the cut0 frame to the cut frame
        n_com: number of principal component used in cos_xt_xv (about 20), ratio_speed_time (20), and procrustes_curve_diff_time (4)
        print_pca_message (bool): whether to print out the variance explained by n_com
        '''
        self.mean_dot, self.err_dot = [], []

        speed_ind = np.argsort(self.speed_list)

        for mode in self.output_mode:
            neural_x = self.neural_x_all[mode].reshape([self.neural_x_all[mode].shape[0], self.neural_x_all[mode].shape[1], -1]) # (n_speed, n_time, features)
            neural_x = neural_x[speed_ind][:, cut0:cut]
            if geo_tool_method == 'cos_xt_xv':
                neural_x = geo_tool.pca_reduce(neural_x, n_components=n_com)
                mean_dot_layer, err_dot_layer = geo_tool.cos_xt_xv(neural_x)
            elif geo_tool_method == 'procrustes_curve_diff_time':
                mean_dot_layer, err_dot_layer = geo_tool.procrustes_curve_diff_time(neural_x, print_message=print_pca_message, n_com=n_com)
            elif geo_tool_method == 'dim_manifold':
                mean_dot_layer, err_dot_layer = geo_tool.dim_manifold(neural_x)
            elif geo_tool_method == 'ratio_speed_time':
                neural_x = geo_tool.pca_reduce(neural_x, n_components=n_com) # processing
                mean_dot_layer, err_dot_layer = geo_tool.ratio_speed_time(neural_x)

            self.mean_dot.append(mean_dot_layer)
            self.err_dot.append(err_dot_layer)
        return self.mean_dot, self.err_dot
