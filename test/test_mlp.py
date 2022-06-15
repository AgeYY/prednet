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
from predusion.mlp_trainer import train
from kitti_settings import *


class Cylinder_dataset(Dataset):
    def __init__(self, length=10000, r=1, transform=None, to_torch=False):
        '''
        transform function acts on the numpy data, then converted to torch if to_torch is true
        '''
        self.length = length
        theta = np.random.uniform(0, 2 * np.pi, n_sample)
        z = np.random.uniform(-1, 1, n_sample)
        X = self.cylinder(theta, z, r=r)

        self.sample = {'X': X, 'theta': theta, 'z': z}

        if transform:
            self.sample = transform(self.sample)

        if to_torch:
            for key in self.sample:
                self.sample[key] = torch.from_numpy(self.sample[key]).type(torch.float)

    def __getitem__(self, idx):
        sample_idx = {}
        for key in self.sample:
            sample_idx[key] = self.sample[key][idx]
        return sample_idx

    def __len__(self):
        return self.length

    @staticmethod
    def cylinder(theta, z, r=1):
        '''
        theta (1d array): angle, unit is rad
        z: same shape as theta
        '''
        fea_map = np.empty((z.shape[0], 3)) # rows are samples, columns are features
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z
        fea_map[:, 0] = x
        fea_map[:, 1] = y
        fea_map[:, 2] = z
        return fea_map

### show the input geometry
#sample = dataset[:]
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
#ax.scatter3D(sample['X'][:, 0], sample['X'][:, 1], sample['X'][:, 2])
#
#plt.show()

INPUT_DIM = 3
OUTPUT_DIM = 2
n_layers = 4 # input, layer 1, ... exclude the output prediction only
tot_title = 'Untrained MLP'

r = 1

train_ratio = 0.8
n_epoch = 20
n_sample = 1000
model_path = os.path.join(WEIGHTS_DIR, 'mlp_cylinder.pt')  # where weights will be saved


dataset = Cylinder_dataset(length=n_sample, r=r, to_torch=True)
train_len = int(len(dataset) * train_ratio)
test_len = len(dataset) - train_len
train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

train_iterator = DataLoader(train_set, batch_size=8, shuffle=True)

# train the MLP
model = MLP(INPUT_DIM, OUTPUT_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
#criterion = loss_cylinder

for i in range(n_epoch):
    train_loss = train(model, train_iterator, optimizer, criterion)
    torch.save(model.state_dict(), model_path)

    print(train_loss)

# visualize the feature map
feamap = model.feature_map(dataset[:]['X'])
theta = dataset[:]['theta'].cpu().detach().numpy()
z = dataset[:]['z'].cpu().detach().numpy()

fig = plt.figure(figsize=(15, 8))

cax1 = fig.add_axes([0.27, 0.95, 0.5, 0.05])
cax2 = fig.add_axes([0.27, 0.05, 0.5, 0.05])

i = 0
for key in feamap:
    if key == 'y_pred':
        continue
    title = 'mlp_cylinder_color_theta_' + key

    ax1 = fig.add_subplot(2, n_layers, i + 1, projection='3d')

    fig, ax = plot_dimension_reduction(feamap[key].cpu().detach().numpy(), method='pca', n_components=3, title=title, colorinfo=theta, fig=fig, ax=ax1, cax=cax1)

    ax2 = fig.add_subplot(2, n_layers, i + 1 + n_layers, projection='3d')
    title = 'mlp_cylinder_color_z_' + key
    fig, ax = plot_dimension_reduction(feamap[key].cpu().detach().numpy(), method='pca', n_components=3, title=title, colorinfo=z, fig=fig, ax=ax2, cax=cax2)

    i += 1

fig.savefig('./figs/' + tot_title + '.pdf')

plt.show()

