import torch
from torch.utils.data import Dataset, DataLoader
import hickle as hkl
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

# general dataset, loading data from file

class Layer_Dataset(Dataset):
    def __init__(self, feamap_path, label_path, label_name_path):
        '''
        feamap (dict): {layer_name: feamap, layer_name: feamap}
        label (np array): [n_observation, n_label]
        label_name (np array str): [n_label]
        '''
        self.feamap = hkl.load(feamap_path)
        self.label = hkl.load(label_path) # label should include the sematic meaning
        self.label_name = hkl.load(label_name_path) # label should include the sematic meaning
        self.length = self.label.shape[0]

    def __getitem__(self, idx):

        obs = {}
        for key in self.feamap:
            obs[key] = self.feamap[key][idx]

        return obs, self.label[idx]

    def __len__(self):
        return self.length

class Meta_Dataset():
    def __init__(self, dataset_name_list, feamap_path_list, label_path_list, label_name_path_list):
        '''
        name_list (list of str): name of the datasets
        label_name_path (list): label name for every dataset must be the same.
        '''
        self.dataset_name_list = dataset_name_list
        self.label_name_path_list = label_name_path_list
        self.n_dataset = len(feamap_path_list)
        self.dataset = []
        for i in range(self.n_dataset):
            self.dataset.append( Layer_Dataset(feamap_path_list[i], label_path_list[i], label_name_path_list[i]) )

    def within_dataset_train_test_validate_split(self, dataset_name, train_ratio, test_ratio):
        '''
        Split the data in one dataset into train test and validate
        '''
        for idx in range(self.n_dataset):
            if dataset_name == self.dataset_name_list: break

        return train_test_validate_split(self.dataset[idx], train_ratio, test_ratio)

    def __getitem__(self, idx):
        '''idx (int or list)'''
        idx_np = np.array([idx]).flatten() # make it as an np array to measure the length
        if len(idx_np) == 1:
            return self.dataset[idx[0]].feamap, self.dataset[idx[0]].label

        else:
            feamap = {}
            for key in self.dataset[0].feamap: # merge each key across different dataset
                feamap[key] = []
                for i in idx_np:
                    feamap[key].append( self.dataset[i].feamap[key] )
                feamap[key] = np.concatenate(feamap[key], axis=0)

            label = []
            for i in idx_np:
                label.append( self.dataset[i].label )
            label = np.concatenate(label, axis=0)

            return feamap, label


def train_test_validate_split(dataset, frac_train, frac_test, random_seed=42):
    '''
    split a dataset into train validate and testset.
    frac_train: 0 to 1
    the frac_validate is 1 - frac_train - frac_test
    '''
    l = len(dataset)
    rand_idx = np.random.permutation(l)
    train_idx, test_idx, validate_idx = np.split(rand_idx, [int(frac_train * l), int((frac_train + frac_test) * l)])
    return dataset[train_idx], dataset[test_idx], dataset[validate_idx]
