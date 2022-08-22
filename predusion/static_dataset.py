import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import hickle as hkl
import numpy as np
import numpy.ma as ma

def replace_nan_mean(data, axis=0):
    '''
    data (array): replace nan entries in the data to the mean value along axis
    '''
    idx = np.isnan(data)
    return np.where(idx, ma.array(data, mask=idx).mean(axis=axis), data)


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
    def __init__(self, feamap_path, label_path, label_name_path, explained_var_thre=None, nan_handle='mean'):
        '''
        feamap (dict): {layer_name: feamap, layer_name: feamap}
        label (np array): [n_observation, n_label]
        label_name (np array str): [n_label]
        explained_var_ratio (float between 0 to 1): preprocessing the feamap using pca
        '''
        self.feamap = hkl.load(feamap_path)

        if not (explained_var_thre is None):
            for key in self.feamap:
                self.feamap[key] = replace_nan_mean(self.feamap[key], axis=0) # average nan neural response to its averaged response across all trails
                self.feamap[key] = self.pca_dim_reduction(self.feamap[key], explained_var_thre)

        self.label = hkl.load(label_path) # label should include the sematic meaning
        self.label = replace_nan_mean(self.label, axis=0) # replace nan label to average label

        self.label_name = hkl.load(label_name_path) # label should include the sematic meaning
        self.length = self.label.shape[0]

    @staticmethod
    def pca_dim_reduction(feamap, explained_var_thre):
        pca = PCA(n_components=None)
        pca.fit(feamap)
        var_explained = np.cumsum(pca.explained_variance_ratio_)
        dim = np.argmax(var_explained>explained_var_thre) + 1
        pca = PCA(n_components=dim)
        return pca.fit_transform(feamap)

    def __getitem__(self, idx):

        obs = {}
        for key in self.feamap:
            obs[key] = self.feamap[key][idx]

        return obs, self.label[idx]

    def __len__(self):
        return self.length

class Meta_Dataset():
    '''arange several datasets'''
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
    rand_idx = np.random.RandomState(seed=random_seed).permutation(l)
    train_idx, test_idx, validate_idx = np.split(rand_idx, [int(frac_train * l), int((frac_train + frac_test) * l)])
    return dataset[train_idx], dataset[test_idx], dataset[validate_idx]

class Toy_Manifold_Dataset():
    @staticmethod
    def manifold(label):
        '''
        input:
          label ([n_observation, 2])
        output:
          feamap ([n_observation, 3])
        '''
        x1, x2 = label[:, 0], label[:, 1]
        y1 = x1
        y2 = 3 * x2
        y3 = x1*x1 + x2*x2
        feamap = np.array([y1, y2, y3]).T
        return feamap

    @staticmethod
    def add_noise(y, scale=0.4):
        return y + np.random.normal(0, scale, y.shape)

    def generate_data(self, train_size, test_size, mean=0, var=1, noise_scale=0.4, test_label_method='linspace'):
        '''var is standard deviation'''
        train_label = np.random.normal(mean, var, [train_size, 2])
        train_feamap = self.manifold(train_label)
        train_feamap = self.add_noise(train_feamap, scale=noise_scale)

        if test_label_method == 'random':
            test_label = np.random.normal(mean, var, [test_size, 2])
        elif test_label_method == 'linspace':
            test_lb_arr = np.linspace(-var, var, test_size)
            test_label_0, test_label_1 = np.meshgrid(test_lb_arr, test_lb_arr, indexing='ij')
            test_label = np.array([test_label_0.flatten(), test_label_1.flatten()]).T

        test_true_feamap = self.manifold(test_label)
        test_feamap = self.add_noise(test_true_feamap, scale=noise_scale)

        return train_label, train_feamap, test_label, test_feamap, test_true_feamap

    @staticmethod
    def tangent_vec(label):
        '''
        input:
          label [n_sample, 2]
        output:
          vec [n_sample, 2, 3]: tangent vectors along two directions. 2 incates two labels, 3 indicates the dim of feamap
        '''
        n_sample = label.shape[0]

        # tangent vector along 0th label
        y1 = np.ones(n_sample)
        y2 = np.zeros(n_sample)
        y3 = 2 * label[:, 0]

        vec = np.array([y1, y2, y3])

        # tangent vector along 0th label
        y1 = np.zeros(n_sample)
        y2 = 3 * np.ones(n_sample)
        y3 = 2 * label[:, 1]

        vec = np.array([vec, np.array([y1, y2, y3])])

        vec = np.transpose(vec, (2, 0, 1) )
        return vec
