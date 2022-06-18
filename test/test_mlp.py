# generate input space data
# make the data as a dataset
# a demo to train the mlp
# feed the input to mlp
# quickly see the geometry of mlp
import os
import numpy as np
import matplotlib.pyplot as plt
from predusion.ploter import plot_dimension_reduction
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from predusion.mlp_agent import MLP
from predusion.mlp_trainer import train, evaluate
from predusion.static_dataset import Surface_dataset
from kitti_settings import *


INPUT_DIM = 3
OUTPUT_DIM = 2
n_layers = 4 # input, layer 1, ... exclude the output prediction only
tot_title = 'Untrained MLP'
#surface = 'cylinder'
surface_shape = 'helicoid'

r = 1
helicoid_alpha, helicoid_beta = 1, 2

weight_decay = 0.001
train_ratio = 0.8
n_epoch = 100
n_sample = 1000
model_path = os.path.join(WEIGHTS_DIR, 'mlp_cylinder.pt')  # where weights will be saved


dataset = Surface_dataset(length=n_sample, to_torch=True, shape=surface_shape)
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
    train_loss = train(model, train_iterator, optimizer, criterion, x_key=dataset.x_key, y_key=dataset.y_key)
    eval_loss= evaluate(model, test_iterator, criterion, x_key=dataset.x_key, y_key=dataset.y_key)
    torch.save(model.state_dict(), model_path)

    if i % 50 == 0:
        print('eval loss: ', eval_loss)
        print('train loss: ', train_loss)

# visualize the feature map
feamap = model.feature_map(dataset[:]['X'])
theta = dataset[:]['lt1'].cpu().detach().numpy()
z = dataset[:]['lt2'].cpu().detach().numpy()

fig = plt.figure(figsize=(15, 8))

cax1 = fig.add_axes([0.27, 0.95, 0.5, 0.05])
cax2 = fig.add_axes([0.27, 0.05, 0.5, 0.05])

i = 0
for key in feamap:
    if key == 'y_pred':
        continue
    title = 'mlp_color_lt1_' + key

    ax1 = fig.add_subplot(2, n_layers, i + 1, projection='3d')

    fig, ax = plot_dimension_reduction(feamap[key].cpu().detach().numpy(), method='pca', n_components=3, title=title, colorinfo=theta, fig=fig, ax=ax1, cax=cax1)

    ax2 = fig.add_subplot(2, n_layers, i + 1 + n_layers, projection='3d')
    title = 'mlp_color_lt2_' + key
    fig, ax = plot_dimension_reduction(feamap[key].cpu().detach().numpy(), method='pca', n_components=3, title=title, colorinfo=z, fig=fig, ax=ax2, cax=cax2)

    i += 1

fig.savefig('./figs/' + tot_title + '.pdf')

plt.show()

