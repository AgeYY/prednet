import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from data_utils import SequenceGenerator

from kitti_settings import *

def real2color(colorinfo):
    pass

from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
def plot_dimension_reduction(data, colorinfo=None, method='mds', n_components=2, title=''):
    '''
    data ([sample, feature])
    '''
    if method=='mds':
        embedding = MDS(n_components=n_components)
    elif method=='lle':
        embedding = LocallyLinearEmbedding(n_components=n_components)
    elif method=='isomap':
        embedding = Isomap(n_components=n_components, n_neighbors=2)
    elif method=='pca':
        embedding = PCA(n_components=n_components)

    print(data.reshape([-1, data.shape[-1]]).shape)
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

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, 'my_X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'my_sources_train.hkl')
output_mode = ['E0', 'E1', 'E2', 'E3']
#output_mode = ['R0', 'R1', 'R2', 'R3']
cut0_time=0 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end. Due to the cutoff, the curvature of artificial video is no longer the same as natural video, but the affect should be minor
cut = 12
embed_method = 'isomap'
n_components = 3
speed_list = np.array([1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9])[:cut]

colorinfo_time = np.arange(cut0_time, cut) # temperal color scheme
colorinfo_time = np.tile(colorinfo_time, (cut, 1)).flatten()
colorinfo_speed = np.tile(speed_list, (cut - cut0_time, 1)).T.flatten()

train_generator = SequenceGenerator(train_file, train_sources, nt, sequence_start_mode='unique', output_mode='prediction')

X_train = train_generator.create_all()

neural_x = X_train[:cut, cut0_time:cut].reshape([cut, cut - cut0_time, -1])

print(colorinfo_time)
# rearange the neural speed
ind = np.argsort(speed_list)
neural_x = neural_x[ind]
plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_time, title=embed_method + '_pixel_color_time')
plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_speed, title=embed_method + '_pixel_color_speed')


neural_x_all = hkl.load(os.path.join(DATA_DIR, 'neural_X_error' + '.hkl'))
for mode in output_mode:
    neural_x = neural_x_all[mode][:cut, cut0_time:cut].reshape([cut, cut - cut0_time, -1]) # (n_speed, n_time, features)
    # rearange the neural speed
    ind = np.argsort(speed_list)
    neural_x = neural_x[ind]

    plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_time, title=embed_method + '_neuron_color_time_{}'.format(mode))
    plot_dimension_reduction(neural_x, method=embed_method, n_components=n_components, colorinfo=colorinfo_speed, title=embed_method + '_neuron_color_speed_{}'.format(mode))


