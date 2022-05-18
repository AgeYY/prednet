# firstly run gen_moving_bar_dataset.py and gen_neural_res_moving_bar.py

import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from data_utils import SequenceGenerator

from kitti_settings import *

from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

def align_data(data):
    '''
    align the mean of different information manifolds to a line
    '''
    delta = 100
    line = np.zeros(data.shape[1:])
    line[:, 0] = delta * np.arange(data.shape[1])
    data_mean = np.mean(data, axis=0)
    delta_data = data_mean - line
    shift_neural_x = np.tile(np.expand_dims(delta_data, axis=0), (12, 1 , 1))
    #shift_neural_x = np.tile(np.expand_dims(neural_x_speed_mean, axis=0), (12, 1 , 1))
    return data - shift_neural_x

def plot_dimension_reduction(data, colorinfo=None, method='mds', n_components=2, title='', n_neighbors=2, align_data=False):
    '''
    data ([sample, feature])
    '''
    if align_data:
        data = align_data(data)
    if method=='mds':
        embedding = MDS(n_components=n_components)
    elif method=='lle':
        embedding = LocallyLinearEmbedding(n_components=n_components)
    elif method=='isomap':
        embedding = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    elif method=='pca':
        embedding = PCA(n_components=n_components)

    data_transformed = embedding.fit_transform(data.reshape([-1, data.shape[-1]]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

    if not (colorinfo is None):
        im = ax.scatter3D(data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2], c=colorinfo, cmap = "viridis", depthshade=False)

    fig.colorbar(im, cax = cax, orientation = 'horizontal')
    plt.title(title)
    plt.savefig('./figs/' + title + '.pdf')

nt = 12 # each prediction group contains nt images
batch_size = 10

out_data_head = 'moving_bar'
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')

output_mode = ['E0', 'E1', 'E2', 'E3']
neural_data_path = 'neural_moving_bar_E' + '.hkl'
#output_mode = ['R0', 'R1', 'R2', 'R3']
#neural_data_path = 'neural_moving_bar_R' + '.hkl'
cut0_time=3 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end. Due to the cutoff, the curvature of artificial video is no longer the same as natural video, but the affect should be minor
cut = 12
embed_method = 'pca'
n_components = 3
speed_list = np.array([1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9])

colorinfo_time = np.arange(cut0_time, cut) # temperal color scheme
colorinfo_time = np.tile(colorinfo_time, (12, 1)).flatten()
colorinfo_speed = np.tile(np.arange(1, 13), (cut - cut0_time, 1)).T.flatten()

train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)

X_train = train_generator.create_all()

neural_x = X_train[:, cut0_time:cut].reshape([12, cut - cut0_time, -1])

# rearange the neural speed
ind = np.argsort(speed_list)
neural_x = neural_x[ind]
plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_time, title=embed_method + '_pixel_color_time')
plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_speed, title=embed_method + '_pixel_color_speed')


neural_x_all = hkl.load(os.path.join(DATA_DIR, neural_data_path))
for mode in output_mode:
    neural_x = neural_x_all[mode][:, cut0_time:cut].reshape([12, cut - cut0_time, -1]) # (n_speed, n_time, features)
    # rearange the neural speed
    ind = np.argsort(speed_list)
    neural_x = neural_x[ind]

    plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_time, title=embed_method + '_neuron_color_time_{}'.format(mode))
    plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_speed, title=embed_method + '_neuron_color_speed_{}'.format(mode))


