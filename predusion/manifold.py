import sys
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import predusion.geo_tool as geo_tool

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
        inv_cov = np.diag(1.0 / 2.0 / np.array(h))

        power = np.einsum(x, [0, 1, 2], inv_cov, [0, 3], x, [3, 1, 2], [1, 2])

        return np.exp(-power) # normalization will always be cancelled out

    def build_kernel(self, query_label, feamap, label):
        '''
        building kernels for fit_by_label
          query_label ( array [n_query_points, n_info] )
          feamap (array [num_sample, num_feature])
          label (array [num_sample, n_labels])
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
        self.label_mesh = self.query_label

        return self.label_mesh.copy(), self.info_manifold.copy()

    def tangent_vector(self, label_id='all'):
        '''
        label_id (int or 'all'): tangent vector of which column of information variables
        0 - n_sample, 1 - n_feature, 2 - n_query, 3 - n_label
        '''
        if type(label_id) == int:
            label_id = [label_id]
        if type(label_id) == str and label_id == 'all':
            lb_diff_select = self.lb_mesh_diff
            kernel_width_select = np.array(self.kernel_width)
        else:
            lb_diff_select = self.lb_mesh_diff[label_id, :, :]
            kernel_width_select = np.array(self.kernel_width)[label_id]

        n_label = lb_diff_select.shape[0]

        xk = np.einsum(self.feamap, [0, 1], self.kernel, [0, 2], [1, 2])
        xkv = np.einsum(self.feamap, [0, 1], self.kernel, [0, 2], lb_diff_select, [3, 0, 2], [2, 3, 1]) # n_query, n_label, n_feature
        k = np.sum(self.kernel, axis=0)
        kv = np.einsum(self.kernel, [0, 2], lb_diff_select, [3, 0, 2], [2, 3])
        term1 = np.einsum(xk, [1, 2], kv, [2, 3], [2, 3, 1])
        term2 = np.einsum(xkv, [2, 3, 1], k, [2], [2, 3, 1])

        kernel_width_reshape = kernel_width_select.reshape((1, n_label, 1))
        vec = (term1 - term2) / (k**2)[..., np.newaxis, np.newaxis] / kernel_width_reshape # the shape is [n_query, n_label, n_features]
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
        n_info = label.shape[1]
        if self.kernel_width is None:
            self.kernel_width = np.ones(n_info)
        query_label = np.array( np.meshgrid(*raw_label_mesh)).transpose().reshape( (-1, n_info) )
        self.build_kernel(query_label, feamap, label)
        return self.fit_by_label()

    def fit_manifold_subspace(self, explained_var_thre):
        '''
        please firstly fit the self.info_manifold
        return:
          dim (int): dimensionality
        '''

        # if self.info_manifold not defined

        try:
            self.info_manifold
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
        ''' Decoding X to the information using the manifold. Score of multiple output would be averaged. Remember to fit manifold firstly
        X (array [n_observations, n_features])
        '''
        pred = self.predict(X)
        score = r2_score(label, pred)

        return score

    def mutual_info_score(self, X, label, sigma=1, normalized=True):
        pred, _ = self.manifold_decoder_score(X, label)
        score = mutual_info.mutual_information_2d(pred, label, sigma=sigma, normalized=normalized)
        return score

#class Layer_manifold():
#    def load_data(self, feamap, label):
#        '''
#        feamap (dict): {'X': [n_observation, n_features], 'R0': [n_observation, n_features], ...}
#        label (array [n_observation, n_labels])
#        '''
#        self.feamap = feamap
#        self.label = label
#        self.num_label = label.shape[1]
#
#        self.ana_group = {key: {} for key in feamap} # create empty group, key indicate layers. this empty dict would be filled in like {(0, 1): Data_manifold} where (0, 1) indicate the combination of label_id
#
#    def fit_info_manifold_grid_all(self, label_mesh, label_id=(0, 1), kernel_name='gaussian', kernel_width=[0.1, 0.1]):
#        '''
#        fit the info_manifold for all keys but single label
#        label_id (int): the ith label
#        '''
#        lb_id_tuple = tuple(label_id)
#        l_mesh = [label_mesh[i] for i in label_id]
#        kw = [kernel_width[i] for i in label_id]
#
#        for key in self.ana_group:
#            self.ana_group[key][lb_id_tuple] = Data_manifold(kernel_width=kw, kernel_name=kernel_name)
#            self.ana_group[key][lb_id_tuple].fit_by_label_grid_mesh(l_mesh, self.feamap[key], self.label[:, lb_id_tuple])
#
#    def manifold_decoder_score_all(self, feamap_test, label_test, label_id=[0, 1]):
#        lb_id_tuple = tuple(label_id)
#        score = {}
#        for key in self.ana_group:
#            score[key] = self.ana_group[key][lb_id_tuple].score(feamap_test[key], label_test[:, lb_id_tuple])
#        return score
#
#    def angle_tangent_vec_all(self, query_label, label_id=[0, 1]):
#        '''
#        calculate the cos of two normalized tangent vectors on the manifold at point query_label.
#        query_label (array [n_query_points, n_info])
#        label_id (list [2]): id of two information variables
#        output:
#          angle: between 0 and 90 degree
#        '''
#
#        angle = {}
#        for key in self.ana_group:
#            vec = self.ana_group[key][lb_id_tuple].tangent_vector(query_label, label_id) # the shape is [n_query, n_label, n_features]
#            vec = vec / np.linalg.norm(vec, axis=2, keepdims=True)
#            vec1, vec2 = vec[:, 0, :], vec[:, 1, :]
#            dot = np.einsum(vec1, [0, 1], vec2, [0, 1], [0])
#            ag = np.arccos(np.clip(dot, -1, 1)) / np.pi * 180
#            angle[key] = ag
#
#        return angle
#
#    def dim_all(self, explained_var_thre, label_id):
#        dim = {} # dimensionality of the manifold
#        for key in self.ana_group:
#            _, dim[key] = self.ana_group[key][tuple(label_id)].fit_manifold_subspace(explained_var_thre)
#        return dim
#
#    def label_dis(self, label_id=None):
#        '''show the histogram of label distribution'''
#        if label_id is None:
#            sns.displot(self.label)
#        else:
#            try:
#                for lid in label_id:
#                    sns.displot(self.label[:, lid])
#            except:
#                sns.displot(self.label[:, label_id])
#        plt.show()
#
#    def search_kernel_width(self, label_mesh, feamap_train, label_train, feamap_validate, label_validate, label_id, kernel_width_list, kernel_name='gaussian'):
#
#        l_mesh = [label_mesh[i] for i in label_id]
#        lb_id_tuple = tuple(label_id)
#
#        score = {}
#        for key in feamap_train:
#            score[key] = []
#            for kw in kernel_width_list:
#                dm = Data_manifold(kernel_width=kw, kernel_name=kernel_name)
#                dm.fit_by_label_grid_mesh(l_mesh, feamap_train[key], label_train[:, label_id])
#                score[key].append(dm.score(feamap_validate[key], label_validate[:, label_id]))
#        return score

class Layer_manifold():
    def __init__(self):
        pass
    def load_data(self, train_feamap, train_label):
        '''
        feamap (dict): {'X': [n_observation, n_features], 'R0': [n_observation, n_features], ...}
        label (array [n_observation, n_labels])
        '''
        self.feamap = train_feamap
        self.label = train_label
        self.num_label = train_label.shape[1]

        self.ana_group = {key: {} for key in train_feamap} # create empty group, key indicate layers. this empty dict would be filled in like {(0, 1): Data_manifold} where (0, 1) indicate the combination of label_id

    def build_kernel(self, query_label, label_id=(0, 1), kernel_width=[0.1, 0.1], kernel_name='gaussian'):
        lb_id_tuple = tuple(label_id)
        kw = [kernel_width[i] for i in label_id]

        for key in self.ana_group:
            self.ana_group[key][lb_id_tuple] = Data_manifold(kernel_width=kw, kernel_name=kernel_name)
            self.ana_group[key][lb_id_tuple].build_kernel(query_label[:, label_id], self.feamap[key], self.label[:, label_id])

    def fit_info_manifold_all(self, label_id):
        '''
        remember to build kernel first
        query_label (array [n_sample, n_label])
        kernel_width (array, n_label): kernel_width for each label_id
        '''
        lb_id_tuple = tuple(label_id)
        for key, agents in self.ana_group.items(): # all layer
            self.ana_group[key][lb_id_tuple].fit_by_label()

    def fit_info_manifold_grid_all(self, label_mesh, label_id=(0, 1), kernel_name='gaussian', kernel_width=[0.1, 0.1]):
        '''
        this function will automatically build kernel.
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

    def tangent_vec_all(self, label_id):
        '''
        This will automatically calculate all the tangent vectors at query_label which is stored when building_kernel. So please remember to build kernel first
        '''
        lb_id_tuple = tuple(label_id)
        vec = {}
        for key in self.ana_group:
                vec[key] = self.ana_group[key][lb_id_tuple].tangent_vector(label_id='all')
        return vec

    def angle_tangent_vec_all(self, label_id=[0, 1], vec_label_id=[0, 1], limit_90=False):
        '''
        calculate the cos of two normalized tangent vectors on the manifold at point query_label.
        query_label (array [n_query_points, n_info])
        label_id (list [n_info]): id for agent. n_info should larger than two
        vec_label_id (list [2]): id along tangent vector
        limit_90 (bool): limit the angle to np.minimum(ag, 180-ag) or not
        output:
          angle: between 0 and 90 degree
        '''
        angle = {}
        lb_id_tuple = tuple(label_id)
        label_id_int = np.array(label_id, dtype=int)

        vec_idx = [np.where(label_id_int == vid) for vid in vec_label_id]
        vec_idx = np.array(vec_idx).flatten()

        for key in self.ana_group:
            vec = self.ana_group[key][lb_id_tuple].tangent_vector(vec_idx) # the shape is [n_query, n_label, n_features]
            vec1, vec2 = vec[:, 0, :], vec[:, 1, :]
            angle[key] = geo_tool.angle_vec(vec1, vec2, limit_90=limit_90)

        return angle

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
