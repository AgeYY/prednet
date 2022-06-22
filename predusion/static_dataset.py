import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Surface_dataset(Dataset):
    def __init__(self, length=10000, shape='cylinder', transform=None, to_torch=False, lt=None, **geo_kward):
        '''
        transform function acts on the numpy data, then converted to torch if to_torch is true
        lt (dict {'lt1': [0, 1, 2, ...], 'lt2': [-0.2, 0, 2.0]}): the whole dataset would be generated from label_data0 = (lt1[0], lt2[0]), label_data1 = (lt1[1], lt2[1])
        '''
        self.length = length

        if shape == 'cylinder':
            if lt is None:
                lt1 = np.random.uniform(0, 2 * np.pi, length)
                lt2 = np.random.uniform(-1, 1, length)
            else:
                lt1 = lt['lt1']
                lt2 = lt['lt2']

            X = self.cylinder(lt1, lt2, **geo_kward)
        elif shape == 'helicoid':
            if lt is None:
                lt1 = np.random.uniform(0, 2 * np.pi, length)
                lt2 = np.random.uniform(-1, 1, length)
            else:
                lt1 = lt['lt1']
                lt2 = lt['lt2']
            X = self.helicoid(lt1, lt2, **geo_kward)

        self.x_key = 'X'
        self.y_key = ['lt1', 'lt2']

        self.sample = {'X': X, 'lt1': lt1, 'lt2': lt2}

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
        fea_map = np.empty((rho.shape[0], 3)) # rows are samples, columns are features

        x = rho * np.cos(alpha * theta)
        y = rho * np.sin(beta * theta)

        fea_map[:, 0] = x
        fea_map[:, 1] = y
        fea_map[:, 2] = theta
        return fea_map

