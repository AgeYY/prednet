# generate input space data
# make the data as a dataset
# a demo to train the mlp
# feed the input to mlp
# quickly see the geometry of mlp
import numpy as np
import matplotlib.pyplot as plt
from predusion.ploter import plot_dimension_reduction
from torch.utils.data import Dataset, DataLoader
import torch
from predusion.mlp_agent import MLP

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

n_sample = 1000
r = 1
dataset = Cylinder_dataset(length=n_sample, r=r, to_torch=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

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

model = MLP(INPUT_DIM, OUTPUT_DIM)

feamap = model.feature_map(dataset[:]['X'])
theta = dataset[:]['theta'].cpu().detach().numpy()
z = dataset[:]['z'].cpu().detach().numpy()

#print(feamap['h1'].shape)
for key in feamap:
    if key == 'y_pred':
        continue
    title = 'mlp_cylinder_color_theta_' + key
    fig, ax = plot_dimension_reduction(feamap[key].cpu().detach().numpy(), method='pca', n_components=3, title=title, colorinfo=theta)

    title = 'mlp_cylinder_color_z_' + key
    fig, ax = plot_dimension_reduction(feamap[key].cpu().detach().numpy(), method='pca', n_components=3, title=title, colorinfo=z)

