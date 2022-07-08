# generate input space data
# make the data as a dataset
# a demo to train the mlp
# feed the input to mlp
# quickly see the geometry of mlp
import os
import numpy as np
import matplotlib.pyplot as plt
from predusion.ploter import plot_dimension_reduction, Ploter_dim_reduction
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from predusion.mlp_agent import MLP
from predusion.mlp_trainer import train, evaluate
from predusion.static_dataset import Surface_dataset
from predusion.geo_tool import angle_PC, angle_between
from predusion.pls_analyzer import PLS_pair
from kitti_settings import *


INPUT_DIM = 3
OUTPUT_DIM = 2
n_layers = 4 # input, layer 1, ... exclude the output prediction only
tot_title = 'Untrained MLP'
#surface_shape = 'cylinder'
surface_shape = 'helicoid'

r = 1
helicoid_alpha, helicoid_beta = 1, 2

weight_decay = 0.001
train_ratio = 0.8
n_epoch = 20
n_sample = 200
model_path = os.path.join(WEIGHTS_DIR, 'mlp_cylinder.pt')  # where weights will be saved
n_component_visu = 2

lt1 = np.random.uniform(0, 2 * np.pi, n_sample)
lt2 = np.random.uniform(-4, 4, n_sample)
label = np.array([lt1, lt2]).T
length = n_sample

dataset = Surface_dataset(length=length, to_torch=True, shape=surface_shape, label=label)
train_len = int(len(dataset) * train_ratio)
test_len = len(dataset) - train_len
train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

train_iterator = DataLoader(train_set, batch_size=128, shuffle=True)
test_iterator = DataLoader(test_set, batch_size=128, shuffle=True)

# train the MLP
model = MLP(INPUT_DIM, OUTPUT_DIM)

optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
criterion = nn.MSELoss()
#criterion = loss_cylinder

for i in range(n_epoch):
    train_loss = train(model, train_iterator, optimizer, criterion)
    eval_loss= evaluate(model, test_iterator, criterion)
    torch.save(model.state_dict(), model_path)

    if i % 50 == 0:
        print('eval loss: ', eval_loss)
        print('train loss: ', train_loss)

# visualize the feature map
feamap = model.feature_map(dataset[:]['X'])
label = dataset[:]['label'].cpu().detach().numpy()
theta, z = label[:, 0], label[:, 1]

i = 0
angle_list, score0_list, score1_list = [], [], []
ploter = Ploter_dim_reduction(method='pca')

#fig = plt.figure(figsize=(15, 8))
for key in feamap:
    if key == 'y_pred':
        continue

    feamap_key = feamap[key].cpu().detach().numpy()
    # calculate the angle
    pls_ana = PLS_pair()
    pls_ana.fit(feamap_key, label)
    angle = pls_ana.angle()
    score = pls_ana.score()

    print(key, angle)
    print(key, score)

    angle_list.append(angle)
    score0_list.append(score[0])
    score1_list.append(score[1])

    ############# visualization
    #title = 'mlp_color_lt1_' + key

    #if n_component_visu == 3:
    #    ax1 = fig.add_subplot(2, n_layers, i + 1, projection='3d')
    #    ax2 = fig.add_subplot(2, n_layers, i + 1 + n_layers, projection='3d')
    #    cax1 = fig.add_axes([0.27, 0.95, 0.5, 0.05])
    #    cax2 = fig.add_axes([0.27, 0.05, 0.5, 0.05])

    #elif n_component_visu == 2:
    #    ax1 = fig.add_subplot(2, n_layers, i + 1)
    #    ax2 = fig.add_subplot(2, n_layers, i + 1 + n_layers)
    #    cax1 = fig.add_axes([0.27, 0.95, 0.5, 0.05])
    #    cax2 = fig.add_axes([0.27, 0.05, 0.5, 0.05])

    #ploter.fit(feamap_key)
    #fig, ax = ploter.plot_dimension_reduction(feamap_key, title=title, colorinfo=theta, fig=fig, ax=ax1, cax=cax1)

    ## show the pls axis
    #def line_i(feamap_key,lt0, scale=1):
    #    pls = PLSRegression(n_components=1)
    #    pls.fit(feamap_key, lt0)
    #    pls_lt0 = pls.x_weights_[:, 0] * scale
    #    origin = np.zeros(pls_lt0.shape)
    #    line0 = np.vstack((origin, pls_lt0))
    #    return line0, pls_lt0

    #lt0 = label[:, [0]]
    #line0, pls_lt0 = line_i(feamap_key, lt0, scale=5)
    #ploter.add_line(line0, ax=ax1, color='b')

    #lt1 = label[:, [1]]
    #line1, pls_lt1 = line_i(feamap_key, lt1, scale=2)
    #ploter.add_line(line1, ax=ax1, color='r')

    #title = 'mlp_color_lt2_' + key
    #fig, ax = ploter.plot_dimension_reduction(feamap_key, title=title, colorinfo=z, fig=fig, ax=ax2, cax=cax2)


    #i += 1

#fig.savefig('./figs/' + tot_title + '.pdf')

#plt.show()

layer = np.arange(len(angle_list))
plt.figure()
plt.title('linear decoding r2 score for the two latent variables')
plt.plot(layer, score0_list[::-1], label='latent0')
plt.plot(layer, score1_list[::-1], label='latent1')
plt.legend()

plt.figure()
plt.title('angle of the two pls')
plt.plot(layer, angle_list[::-1])
plt.ylim([0, 90])
plt.show()

