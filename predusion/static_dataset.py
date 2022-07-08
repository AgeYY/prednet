import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Surface_dataset(Dataset):
    def __init__(self, length=10000, shape='cylinder', transform=None, to_torch=False, label=None, **geo_kward):
        '''
        transform function acts on the numpy data, then converted to torch if to_torch is true
        lt (dict {'lt1': [0, 1, 2, ...], 'lt2': [-0.2, 0, 2.0]}): the whole dataset would be generated from label_data0 = (lt1[0], lt2[0]), label_data1 = (lt1[1], lt2[1])
        '''
        self.length = length

        if label is None:
            if shape == 'cylinder':
                lt1 = np.random.uniform(0, 2 * np.pi, length)
                lt2 = np.random.uniform(-1, 1, length)
            elif shape == 'helicoid':
                lt1 = np.random.uniform(0, 2 * np.pi, length)
                lt2 = np.random.uniform(-1, 1, length)
            label = np.array([lt1, lt2]).T

        if shape == 'cylinder':
            X = self.cylinder(label[:, 0], label[:, 1], **geo_kward)
        elif shape == 'helicoid':
            X = self.helicoid(label[:, 0], label[:, 1], **geo_kward)

        self.sample = {'X': X, 'label': label}

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

        fea_map[:, 0] = x
        fea_map[:, 1] = y
        fea_map[:, 2] = z
        return fea_map


    @staticmethod
    def helicoid(theta, rho, alpha=1, beta=1):
        '''
        theta, rho (array [n_sample])
        '''
        fea_map = np.empty((rho.shape[0], 3)) # rows are samples, columns are features

        x = rho * np.cos(alpha * theta)
        y = rho * np.sin(beta * theta)

        fea_map[:, 0] = x
        fea_map[:, 1] = y
        fea_map[:, 2] = theta
        return fea_map

