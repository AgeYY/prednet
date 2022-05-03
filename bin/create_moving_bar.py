from predusion.immaker import Moving_square

width = 200
step = 12
speed_list = range(0, 12) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image
init_pos = [30, width//2]

# create raw video
ms = Moving_square(width=width)
for sp in speed_list:
    ms.create_video(init_pos=init_pos, speed=sp, step=step)
    ms.save_image(save_dir_label='moving_bar/sp_' + str(sp) + '/')
    ms.clear_image()

# process into data
# ...

# Run the data to the prednet
import os
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
import hickle as hkl
import matplotlib.pyplot as plt

from predusion.agent import Agent
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

nt = 12 # each prediction group contains nt images
batch_size = 10

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, 'my_X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'my_sources_train.hkl')
output_mode = ['E0', 'E1', 'E2', 'E3']
cutoff=2 # the curvature of a trajectory is the mean from curvature from the cutoff frame to the end. Due to the cutoff, the curvature of artificial video is no longer the same as natural video, but the affect should be minor

train_generator = SequenceGenerator(train_file, train_sources, nt, sequence_start_mode='unique', output_mode='prediction')

X_train = train_generator.create_all()

## check the video
#one_video = X_train[-2]
#for im in one_video:
#    plt.imshow(im)
#    plt.show()

sub = Agent()
sub.read_from_json(json_file, weights_file)

output = sub.output_multiple(X_train, output_mode=output_mode, batch_size=batch_size)
print(output)

hkl.dump(output, os.path.join(DATA_DIR, 'neural_X_error' + '.hkl'))

from sklearn.decomposition import PCA

cut = 12
n_com = None
#neural_x_all = hkl.load(os.path.join(DATA_DIR, 'neural_X' + '.hkl'))
neural_x_all = hkl.load(os.path.join(DATA_DIR, 'neural_X_error' + '.hkl'))

mean_dot, sem_dot = [], []
#speed_list = np.array([0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])
#
#from sklearn.manifold import MDS
#def dimension_reduction(data, method='mds'):
#    '''
#    data ([sample, feature])
#    '''
#    embedding = MDS(n_components=2)
#
#    data_transformed = embedding.fit_transform(data)
#
#    plt.figure()
#    plt.scatter(data_transformed[:, 0], data_transformed[:, 1])
#    plt.show()

for mode in output_mode:
    neural_x = neural_x_all[mode][:cut, :cut].reshape([cut, cut, -1]) # (n_speed, n_time, features)
    pca = PCA(n_components=n_com)
    neural_x = pca.fit_transform(neural_x.reshape([cut*cut, -1])).reshape([cut, cut, -1])


    speed_list = np.array([0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])[:cut]

    # rearange the neural speed
    ind = np.argsort(speed_list)
    neural_x = neural_x[ind]

    # calculate the different along the temporal direction
    neural_x_time = np.diff(neural_x, axis=1)[1:]
    neural_x_speed = np.diff(neural_x, axis=0)[:, 1:]

    # calculate the cos
    dot = np.tensordot(neural_x_time, neural_x_speed, axes=([2], [2])) / np.linalg.norm(neural_x_time, axis=2) / np.linalg.norm(neural_x_speed, axis=2)

    dot_flat = dot.flatten()
    
    mean_dot.append(np.mean(dot_flat))
    sem_dot.append(np.std(dot_flat) / np.sqrt(np.size(dot_flat)))


neural_x = X_train[:cut, :cut].reshape([cut, cut, -1])
pca = PCA(n_components=n_com)
neural_x = pca.fit_transform(neural_x.reshape([cut*cut, -1])).reshape([cut, cut, -1])

speed_list = np.array([0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])[:cut]

# rearange the neural speed
ind = np.argsort(speed_list)
neural_x = neural_x[ind]

# calculate the different along the temporal direction
neural_x_time = np.diff(neural_x, axis=1)[1:]
neural_x_speed = np.diff(neural_x, axis=0)[:, 1:]

# calculate the cos
dot = np.tensordot(neural_x_time, neural_x_speed, axes=([2], [2])) / np.linalg.norm(neural_x_time, axis=2) / np.linalg.norm(neural_x_speed, axis=2)

dot_flat = dot.flatten()
    
mean_dot.insert(0, np.mean(dot_flat))
sem_dot.insert(0, np.std(dot_flat) / np.sqrt(np.size(dot_flat)))
#std_dot = np.std(dot_flat)
print(mean_dot, sem_dot)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(range(-1, 4), mean_dot)
plt.errorbar(range(-1, 4), mean_dot, yerr=sem_dot)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Layer of the Prednet \n -1 means the pixels')
plt.ylabel('cos of the angle between the tangent vector along time and speed')
plt.show()
