import sys
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import copy
from scipy.stats import multivariate_normal

class Data_manifold():

    def __init__(self, kernel_width=None, kernel_name='gaussian'):
        '''
        kernel_width is sigma square
        '''
        self.kernel_dic = {'bin': self.bin_kernel, 'gaussian': self.gaussian_kernel}
        self.kernel_width = kernel_width
        self.kernel_name = kernel_name

    @staticmethod
    def bin_kernel(u, h=0.1):
        return np.abs(u) <= h / 2

    @staticmethod
    def gaussian_kernel(u, h=0.1):
        return np.exp( - u**2 / 2.0 / h)

    @staticmethod
    def mul_gaussian_kernel(x, h=None):

        dim = len(h)
        inv_cov = np.diag(1.0 / 2.0 / np.array(h))

        power = np.einsum(x, [0, 1, 2], inv_cov, [0, 3], x, [3, 1, 2], [1, 2])

        return np.exp(-power) # normalization will always be cancelled out

    def build_kernel(self, query_label, feamap, label):
        '''
        building kernels for fit_by_label
          query_label ( array [n_query_points, n_info] )
          feamap (array [num_sample, num_feature])
          label (array [num_sample, 2])
        '''
        self.query_label = query_label
        self.feamap = feamap
        self.label = label

        if self.kernel_width is None:
            n_info = self.label.shape[1]
            self.kernel_width = np.ones(n_info)
        n_label = query_label.shape[1] # number of layers, also equal to the intrinsic dim of the manifold

        self.kernel = [] # create kernel for every label

        lb_mesh_diff = []
        for i in range(n_label):
            lb_meshx, lb_meshy = np.meshgrid(query_label[:, i], label[:, i])
            lb_mesh_diff.append(lb_meshx - lb_meshy)
        self.lb_mesh_diff = np.array(lb_mesh_diff) # [n_info, n_sample, n_query]
        self.kernel = self.mul_gaussian_kernel(self.lb_mesh_diff, h=self.kernel_width) # [n_sample, n_query]

    def fit_by_label(self):
        '''fit the manifold, remember to first build the kernel'''
        kernel_norm = np.sum(self.kernel, axis=0)
        self.info_manifold = self.kernel.T @ self.feamap / kernel_norm[..., np.newaxis]
        self.label_mesh = self.query_label.copy()

        return self.label_mesh.copy(), self.info_manifold.copy()

    def tangent_vector(self, label_id=0):
        '''
        label_id (int or 'all'): tangent vector of which column of information variables
        '''
        # 0 - n_sample, 1 - n_feature, 2 - n_query
        xk = np.einsum(self.feamap, [0, 1], self.kernel, [0, 2], [1, 2])
        xkv = np.einsum(self.feamap, [0, 1], self.kernel, [0, 2], self.lb_mesh_diff[label_id], [0, 2], [2, 1])
        k = np.sum(self.kernel, axis=0)
        kv = np.einsum(self.kernel, [0, 2], self.lb_mesh_diff[label_id], [0, 2], [2])
        term1 = np.einsum(xk, [1, 2], kv, [2], [2, 1])
        term2 = np.einsum(xkv, [2, 1], k, [2], [2, 1])
        vec = 1.0 / self.kernel_width[label_id] * (term1 - term2) / (k**2)[..., np.newaxis]
        return vec

    def fit_by_label_grid_mesh(self, raw_label_mesh, feamap, label):
        '''
        average feature values with similar label, which is called as info_manifold. slightly faster than fit_by_label. recommend to use fit_by_label
        input:
          feamap (array [num_sample, num_feature])
          label (array [num_sample, 2])
        output:
          self.label_mesh, self.info_manifold
        '''
        if self.kernel_width is None:
            n_info = label.shape[1]
            self.kernel_width = np.ones(n_info)

        n_label = len(raw_label_mesh) # number of layers, also equal to the intrinsic dim of the manifold

        kernel = [] # create kernel for every label
        for i in range(n_label):
            lb_meshx, lb_meshy = np.meshgrid(raw_label_mesh[i], label[:, i])
            lb_mesh_diff = lb_meshx - lb_meshy
            kernel.append( self.kernel_dic[self.kernel_name](lb_mesh_diff, h=self.kernel_width[i]) )

        kernel_norm_command = [] # denominator
        for i in range(n_label):
            kernel_norm_command.append(kernel[i])
            kernel_norm_command.append([0, i+1])
        end = [i+1 for i in range(n_label)]
        kernel_norm_command.append(end)

        kernel_norm = np.einsum(*kernel_norm_command)
        # the resulting command looks like: kernel_norm = np.einsum(kernel[0], [0, 1], kernel[1], [0, 2], [1, 2])

        info_manifold_command = kernel_norm_command # just rename it. Ready to calculate numerator
        info_manifold_command.pop()
        info_manifold_command.append(feamap)
        info_manifold_command.append([0, n_label+1])

        end = [i+1 for i in range(n_label+1)]
        info_manifold_command.append(end)

        self.info_manifold = np.einsum(*info_manifold_command) / kernel_norm[..., np.newaxis]
        # the resulting command looks like: self.info_manifold = np.einsum(kernel[0], [0, 1], kernel[1], [0, 2], feamap, [0, 3], [1, 2, 3]) / kernel_mn_norm[..., np.newaxis]
        self.label_mesh = np.array( np.meshgrid(*raw_label_mesh) ).transpose()

        self.info_manifold = self.info_manifold.reshape( (-1, feamap.shape[1]) )
        self.label_mesh = self.label_mesh.reshape( (-1, n_label) )

        return self.label_mesh.copy(), self.info_manifold.copy()

    def fit_manifold_subspace(self, explained_var_thre):
        '''
        please firstly fit the self.info_manifold
        return:
          dim (int): dimensionality
        '''

        # if self.info_manifold not defined
        try: self.info_manifold
        except NameError:
            print('Please fit the information manifold first\n')
            sys.exit()

        pca = PCA(n_components=None)
        pca.fit(self.info_manifold)
        var_explained = np.cumsum(pca.explained_variance_ratio_)
        self.dim = np.argmax(var_explained>explained_var_thre) + 1
        self.pca = PCA(n_components=self.dim)
        self.pca.fit(self.info_manifold)
        return self.pca, self.dim

    def manifold(self, label_query, feamap, label, kernel_name='gaussian', kernel_width=[0.5, 0.5]):
        '''
        output a vector on the manifold which encode label_query. Same as fit_info_manifold, less efficient but more flexiable
        label_query (array (n_query_sample, 2))
        '''
        pass

    def predict(self, X):
        '''
        predict the label of X
        input:
          X (array [n_observations, n_features])
        output:
          pred (array [n_observations, n_labels])
        '''
        pred = []
        for obs in X:
            distance = np.linalg.norm(obs - self.info_manifold, axis=-1)
            pred.append( self.label_mesh[np.argmin(distance)] )

        pred = np.array(pred)
        return pred

    def score(self, X, label):
        ''' Decoding X to the information using the manifold. Score of multiple output would be averaged
        X (array [n_observations, n_features])
        '''
        pred = self.predict(X)
        score = r2_score(label, pred)

        return score

    def mutual_info_score(self, X, label, sigma=1, normalized=True):
        pred, _ = self.manifold_decoder_score(X, label)
        score = mutual_info.mutual_information_2d(pred, label, sigma=sigma, normalized=normalized)
        return score

class Layer_manifold():
    def load_data(self, feamap, label):
        '''
        feamap (dict): {'X': [n_observation, n_features], 'R0': [n_observation, n_features], ...}
        label (array [n_observation, n_labels])
        '''
        self.feamap = feamap
        self.label = label
        self.num_label = label.shape[1]

        self.ana_group = {key: {} for key in feamap} # create empty group, key indicate layers. this empty dict would be filled in like {(0, 1): Data_manifold} where (0, 1) indicate the combination of label_id

    def fit_info_manifold_all(self, label_mesh, label_id=(0, 1), kernel_name='gaussian', kernel_width=[0.1, 0.1]):
        '''
        fit the info_manifold for all keys but single label
        label_id (int): the ith label
        '''
        lb_id_tuple = tuple(label_id)
        l_mesh = [label_mesh[i] for i in label_id]
        kw = [kernel_width[i] for i in label_id]

        for key in self.ana_group:
            self.ana_group[key][lb_id_tuple] = Data_manifold(kernel_width=kw, kernel_name=kernel_name)
            self.ana_group[key][lb_id_tuple].fit_by_label_grid_mesh(l_mesh, self.feamap[key], self.label[:, lb_id_tuple])

    def manifold_decoder_score_all(self, feamap_test, label_test, label_id=[0, 1]):
        lb_id_tuple = tuple(label_id)
        score = {}
        for key in self.ana_group:
            score[key] = self.ana_group[key][lb_id_tuple].score(feamap_test[key], label_test[:, lb_id_tuple])
        return score

    def dim_all(self, explained_var_thre, label_id):
        dim = {} # dimensionality of the manifold
        for key in self.ana_group:
            _, dim[key] = self.ana_group[key][tuple(label_id)].fit_manifold_subspace(explained_var_thre)
        return dim

    def label_dis(self, label_id=None):
        '''show the histogram of label distribution'''
        if label_id is None:
            sns.displot(self.label)
        else:
            try:
                for lid in label_id:
                    sns.displot(self.label[:, lid])
            except:
                sns.displot(self.label[:, label_id])
        plt.show()

    def search_kernel_width(self, label_mesh, feamap_train, label_train, feamap_validate, label_validate, label_id, kernel_width_list, kernel_name='gaussian'):

        l_mesh = [label_mesh[i] for i in label_id]
        lb_id_tuple = tuple(label_id)

        score = {}
        for key in feamap_train:
            score[key] = []
            for kw in kernel_width_list:
                dm = Data_manifold(kernel_width=kw, kernel_name=kernel_name)
                dm.fit_by_label_grid_mesh(l_mesh, feamap_train[key], label_train[:, label_id])
                score[key].append(dm.score(feamap_validate[key], label_validate[:, label_id]))
        return score
