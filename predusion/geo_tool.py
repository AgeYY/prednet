import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge as scikit_ridge
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from predusion import mutual_info

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Code contributed from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / np.pi * 180
    return np.minimum(angle, 180-angle)

def shift_mean(neural_x, avg_axis=0):
    '''
    shift the mean of manifolds to 0
    '''
    n_manifold = neural_x.shape[avg_axis]
    mean_manifold = np.mean(neural_x, axis=avg_axis) # array, each element is the mean of speed manifold across different speeds at a particular time point
    mean_manifold = np.expand_dims(mean_manifold, axis=avg_axis)
    repeat_list = [1] * len(neural_x.shape)
    repeat_list[avg_axis] = n_manifold
    mean_manifold = np.tile(mean_manifold, repeat_list)
    return neural_x - mean_manifold

def angle_PC(neural_x, label, error_bar='std'):
    '''
    nerual_x ([n_sample, n_feature]): neural response
    label ([n_sample, n_labels = 2]):
    '''
    neural_x_train, neural_x_test, label_train, label_test = train_test_split(neural_x, label, test_size=0.3)

    lt0_train = label_train[:, [0]]
    lt1_train = label_train[:, [1]]

    pls = PLSRegression(n_components=1)
    pls.fit(neural_x_train, lt0_train)

    score = pls.score(neural_x_test, label_test[:, [0]])
    print('score: ', score)

    pls_lt0 = pls.x_weights_[:, 0]

    pls.fit(neural_x_train, lt1_train)
    pls_lt1 = pls.x_weights_[:, 0]

    return angle_between(pls_lt0, pls_lt1), 0

def pls_angle_score(neural_x, label):
    '''
    nerual_x ([n_sample, n_feature]): neural response
    label ([n_sample, n_labels = 2]):
    '''
    neural_x_train, neural_x_test, label_train, label_test = train_test_split(neural_x, label, test_size=0.3)

    lt0_train = label_train[:, [0]]
    lt1_train = label_train[:, [1]]

    pls = PLSRegression(n_components=1)
    pls.fit(neural_x_train, lt0_train)

    score_lt0 = pls.score(neural_x_test, label_test[:, [0]])
    score_lt1 = pls.score(neural_x_test, label_test[:, [1]])

    pls_lt0 = pls.x_weights_[:, 0]

    pls.fit(neural_x_train, lt1_train)
    pls_lt1 = pls.x_weights_[:, 0]

    return angle_between(pls_lt0, pls_lt1), score_lt0, score_lt1

#def angle_PC_grid(neural_x, error_bar='std'):
#
#    #neural_x = np.random.normal(scale=1, size=neural_x.shape)
#    label_speed = np.arange(neural_x.shape[0])
#    label_time = np.arange(neural_x.shape[1])
#    label_speed, label_time = np.meshgrid(label_speed, label_time)
#    label_speed, label_time = label_speed.T, label_time.T
#
#    neural_x_flat = neural_x.reshape((-1, neural_x.shape[-1]))
#    label_speed = label_speed.reshape((-1, 1))
#    label_time = label_time.reshape((-1, 1))
#
#    pls = PLSRegression(n_components=1)
#    pls.fit(neural_x_flat, label_speed)
#    speed_pls = pls.x_weights_[:, 0]
#
#    pls = PLSRegression(n_components=1)
#    pls.fit(neural_x_flat, label_time)
#    time_pls = pls.x_weights_[:, 0]
#
#    return angle_between(time_pls, speed_pls), 0
#
#    #neural_x_proj = np.tensordot(neural_x, speed_pls, axes=([-1, -1]))
#    ##neural_x_proj = np.tensordot(neural_x, time_pls, axes=([-1, -1]))
#    ##print(neural_x_proj.shape)
#    #print(neural_x_proj)
#    #e_v_var_t = np.mean( np.var(neural_x_proj, axis=1) )
#    #e_t_var_v = np.mean( np.var(neural_x_proj, axis=0) )
#
#    #return e_v_var_t / e_t_var_v, 0
#    ##return e_t_var_v / e_v_var_t, 0
#
#    #neural_x_time = shift_mean(neural_x, avg_axis=1)
#    #neural_x_speed = shift_mean(neural_x, avg_axis = 0)
#    ##print(neural_x_time.shape)
#
#    ## collecting all tangent vectors
#    #neural_x_time_flat = neural_x_time.reshape((-1, neural_x_time.shape[-1]))
#    #neural_x_speed_flat = neural_x_speed.reshape((-1, neural_x_speed.shape[-1]))
#
#    #pca = PCA(n_components=1)
#    #pca.fit_transform(neural_x_time_flat)
#    #time_pc = pca.components_[0]
#    #pca.fit_transform(neural_x_speed_flat)
#    #speed_pc = pca.components_[0]
#
#    #return angle_between(time_pc, speed_pc), 0

def var_on_pls_grid(neural_x, label_0, label_1, error_bar='std'):
    '''
    nerual_x ([n_sample, n_feature]): neural response
    '''
    lt0 = label[:, [0]]

    pls = PLSRegression(n_components=1)
    pls.fit(neural_x, lt0)

    pls_lt0 = pls.x_scores_[:, 0].flatten()
    lt0 = lt0.flatten()
    pass

def cos_xt_xv(neural_x, error_bar='std'):
    '''
    calculate the mean and se (or std) of one layer. The cos value of two tangent vectors in the same point
    input:
      neural_x ([n_speed, n_time, feature])
    '''
    # calculate the different along the temporal direction
    neural_x_time = np.gradient(neural_x, axis=1)
    neural_x_speed = np.gradient(neural_x, axis=0)

    # collecting all tangent vectors
    neural_x_time_flat = neural_x_time.reshape((-1, neural_x_time.shape[-1]))
    neural_x_speed_flat = neural_x_speed.reshape((-1, neural_x_speed.shape[-1]))
    ## fit the subspace
    #nc = 6
    #pca = PCA(n_components=nc)
    #neural_x_speed_flat = pca.inverse_transform( pca.fit_transform(neural_x_speed_flat) )
    #neural_x_speed = neural_x_speed_flat.reshape( neural_x_speed.shape )
    #print('speed tangent: ', np.cumsum(pca.explained_variance_ratio_))
    #neural_x_time_flat = pca.inverse_transform( pca.fit_transform(neural_x_time_flat) )
    #neural_x_time = neural_x_time_flat.reshape( neural_x_time.shape )
    #print('time tangent: ', np.cumsum(pca.explained_variance_ratio_))

    # calculate the cos
    dot = np.sum(neural_x_time * neural_x_speed, axis=-1) / np.linalg.norm(neural_x_time, axis=2) / np.linalg.norm(neural_x_speed, axis=2)
    #print(np.isnan(dot))

    dot_flat = dot.flatten()

    if error_bar=='std':
        err = np.std(dot_flat)
    else: # sem
        err = np.std(dot_flat) / np.sqrt(np.size(dot_flat))

    return np.mean(dot_flat), err

def cos_para_layer(neural_x, error_bar='std'):
    '''
    calculate the mean and se (or std) of one layer
    input:
      neural_x ([n_speed, n_time, feature])
    '''
    # calculate the different along the temporal direction
    #neural_x_speed = np.gradient(neural_x, axis=0)
    neural_x_speed_mean = np.mean(neural_x, axis=0)
    #shift_neural_x = np.tile(np.expand_dims(neural_x[0], axis=0), (cut, 1, 1))
    shift_neural_x = np.tile(np.expand_dims(neural_x_speed_mean, axis=0), (cut, 1 , 1))
    #neural_x_speed = neural_x - np.tile(neural_x[:, 0])
    neural_x_speed = neural_x - shift_neural_x
    #neural_x_speed = neural_x_speed[1:, :]

    nt = neural_x_speed.shape[1]
    # calculate the cos
    dot = []

    for i in range(nt - 1):
        for j in range(i + 1, nt):
            dot.append( np.sum(neural_x_speed[:, i] * neural_x_speed[:, j], axis=-1) / np.linalg.norm(neural_x_speed[:, i], axis=-1) / np.linalg.norm(neural_x_speed[:, j], axis=-1) )
            #dot.append(frdist(neural_x_speed[:, i], neural_x_speed[:, j]))

    dot_flat = np.array(dot).flatten()

    if error_bar=='std':
        err = np.std(dot_flat)
    else: # sem
        err = np.std(dot_flat) / np.sqrt(np.size(dot_flat))

    return np.mean(dot_flat), err

def pca_reduce(neural_x, n_components=None, print_message=False, with_var_explained=False):
    '''
    reduce the dimensionality using pca
    input:
      neural_x ([n_speed, n_time, feature])
    output:
      neural_x ([n_speed, n_time, n_components])
      total_var_explained (float)
    '''
    if n_components is None:
        return neural_x

    # pca processing
    neural_x_flat = neural_x.reshape(-1, neural_x.shape[-1])
    pca = PCA(n_components=n_components)
    neural_x_flat = pca.fit_transform(neural_x_flat)
    neural_x = neural_x_flat.reshape(neural_x.shape[0], neural_x.shape[1], n_components)
    if print_message:
        print('the total variance explained is: ', np.cumsum(pca.explained_variance_ratio_)[-1])
    if with_var_explained:
        return neural_x, np.cumsum(pca.explained_variance_ratio_)

    return neural_x

def trans_norm(curve):
    mean = np.mean(curve, axis=0)
    mean = np.tile(mean[np.newaxis, ...], (curve.shape[0], 1))
    var = np.sum(np.var(curve, axis=0))
    curve_transformed = (curve - mean) / np.sqrt(var)
    return curve_transformed

def procrustes_translate_normalize(curve1, curve2):
    '''
    curve ([n_points, n_dim])
    '''
    curve1_t = trans_norm(curve1)
    curve2_t = trans_norm(curve2)
    d = (curve1_t - curve2_t)**2
    return np.sum(d.flatten())

def procrustes_curve_diff_time(neural_x, error_bar='std', n_com=3, print_message=False, method=None):
    '''
    calculate the disparity between curves x(t1, v) and x(t2, v)
    input:
      neural_x ([n_speed, n_time, feature])
    '''

    disparity = []

    for i in range(neural_x.shape[1] - 1):
        for j in range(i + 1, neural_x.shape[1]):
            neural_x_pair = neural_x[:, [i, j], :]
            neural_x_pair = pca_reduce(neural_x_pair, n_components=n_com, print_message=print_message)
            if method == 'tran_norm':
                disparity_temp = procrustes_translate_normalize(neural_x_pair[:, 0], neural_x_pair[:, 1])

            else:
                _, _, disparity_temp = procrustes(neural_x_pair[:, 0], neural_x_pair[:, 1])
            disparity.append(disparity_temp)

    if error_bar=='std':
        err = np.std(disparity)
    else: # sem
        err = np.std(disparity) / np.sqrt(np.size(disparity))

    return np.mean(disparity), err

def dim_manifold(neural_x, error_bar='std', n_com_max=None, print_message=False, thresh=0.95):
    '''
    dimensionality of the information manifold
    '''
    if n_com_max is None:
        n_com_max = neural_x.shape[0]
    pca = PCA(n_components=n_com_max)
    dim = []
    for i in range(neural_x.shape[1]):
        pca.fit(neural_x[:, i])
        var_explained = np.cumsum(pca.explained_variance_ratio_)

        for j, var_exp in enumerate(var_explained):
            if var_exp > thresh: break
        dim.append(j + 1)

    if error_bar=='std':
        err = np.std(dim)
    else: # sem
        err = np.std(dim) / np.sqrt(np.size(dim))

    return np.mean(dim), err

def ratio_speed_time(neural_x, error_bar='std', n_com=None, print_message=False):
    '''
    calculate the mean and se (or std) of one layer
    input:
      neural_x ([n_speed, n_time, feature])

    '''
    var_time, var_speed = [], []
    # variance of time
    for i_sp in range(neural_x.shape[0]):
        neural_x_time = neural_x[i_sp]
        var_time_temp = np.sum(np.var(neural_x_time, axis=0))
        var_time.append(var_time_temp)
    for i_t in range(neural_x.shape[1]):
        neural_x_speed = neural_x[:, i_t]
        var_speed_temp = np.sum(np.var(neural_x_speed, axis=0))
        var_speed.append(var_speed_temp)
    std_t, std_s = np.std(var_time), np.std(var_speed)
    mean_var_t, mean_var_s = np.mean(var_time), np.mean(var_speed)
    ratio = mean_var_s / mean_var_t
    std_ratio = ratio * np.sqrt((std_s / mean_var_s)**2 + (std_t / mean_var_t)**2) # propagation of error

    return ratio, std_ratio

class Single_geo_analyzer():
    def __init__(self):
        self.kernel_dic = {'bin': self.bin_kernel, 'gaussian': self.gaussian_kernel}

    @staticmethod
    def bin_kernel(u, h=0.1):
        return np.linalg.norm(u) <= h / 2

    @staticmethod
    def gaussian_kernel(u, h=0.1):
        return np.exp( - u**2 / 2.0 / h) / np.sqrt(2 * np.pi * h)

    @staticmethod
    def pls_dim_reduction(feamap, label):
        ''' reduce the dimension to have better estimation on the manifold and Sigma'''
        pass

    def fit_info_manifold(self, label_mesh, feamap, label, kernel_name='gaussian', kernel_width=0.1):
        '''
        average feature values with similar label, which is called as info_manifold
        input:
          label_mesh (array): label_mesh for the fitted info_manifold
          feamap (array [num_sample, num_feature])
          label (array [num_sample])
        output:
          self.label_mesh, self.info_manifold
        '''
        sample_size = feamap.shape[0]
        mesh_size = label_mesh.shape[0]
        kernel_mat = np.empty((mesh_size, sample_size))

        for i, li in enumerate(label_mesh):
            kernel_mat[i] = self.kernel_dic[kernel_name](li - label, h=kernel_width)

        kernel_norm = kernel_mat / kernel_mat.sum(axis=1, keepdims=1)

        self.info_manifold = np.dot(kernel_norm, feamap)
        self.label_mesh = label_mesh

        # fit the Sigma

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


    def linear_regression_score(self, explained_var_thre, feamap_train, label_train, feamap_test, label_test):
        '''
        feamap_train ([n_observation, n_features]):
        label_train (n_observation): only one label
        return:
          the amount of information encoded by the subspace of the manifold, as measured by: 1. find the pca space expalins explained_var_thre 2. linear regression 3. test coefficient of determination of the prediction
        '''
        self.fit_manifold_subspace(explained_var_thre)
        feamap_proj_train, feamap_proj_test = self.pca.transform(feamap_train), self.pca.transform(feamap_test)
        self.clf = scikit_ridge()
        self.clf.fit(feamap_proj_train, label_train)
        pred = self.clf.predict(feamap_proj_test)
        self.score = self.clf.score(feamap_proj_test, label_test)
        return pred, self.score

    def manifold_decoder_score(self, X, label=None):
        ''' Decoding X to the information using the manifold
        X (array [n_observations, n_features])
        '''
        pred = []
        for obs in X:
            distance = np.linalg.norm(obs - self.info_manifold, axis=1)
            pred.append( self.label_mesh[np.argmin(distance)] )

        pred = np.array(pred)
        if label is None:
            self.score = 0
        else:
            self.score = r2_score(label, pred)

        return pred, self.score

    def mutual_info_score(self, X, label, sigma=1, normalized=True):
        pred, _ = self.manifold_decoder_score(X, label)
        score = mutual_info.mutual_information_2d(pred, label, sigma=sigma, normalized=normalized)
        return score

class Double_geo_analyzer(Single_geo_analyzer):
    def fit_info_manifold(self, label_mesh, feamap, label, kernel_name='gaussian', kernel_width=[0.1, 0.1]):
        '''
        average feature values with similar label, which is called as info_manifold
        input:
          label_mesh ([label_mesh0, label_mesh1]): the final label mesh would be the matrix product of these two
          feamap (array [num_sample, num_feature])
          label (array [num_sample])
        output:
          self.label_mesh, self.info_manifold
        '''
        sample_size = feamap.shape[0]
        label_mesh0, label_mesh1 = label_mesh

        kernel_mat0 = np.empty((label_mesh0.shape[0], sample_size))
        kernel_mat1 = np.empty((label_mesh1.shape[0], sample_size))

        for m, li in enumerate(label_mesh0):
            kernel_mat0[m] = self.kernel_dic[kernel_name](li - label[:, 0], h=kernel_width[0])

        for n, li in enumerate(label_mesh1):
            kernel_mat1[n] = self.kernel_dic[kernel_name](li - label[:, 1], h=kernel_width[1])

        kernel_mn_norm = kernel_mat0 @ kernel_mat1.T

        self.info_manifold = np.empty( (label_mesh0.shape[0], label_mesh1.shape[0], feamap.shape[1]) )
        self.label_mesh = np.empty((label_mesh0.shape[0], label_mesh1.shape[0], 2))

        for m in range(label_mesh0.shape[0]):
            for n in range(label_mesh1.shape[0]):
                for i in range(feamap.shape[0]):
                    temp = feamap[i] * kernel_mat0[m, i] * kernel_mat1[n, i]
                #self.info_manifold[m, n] = temp / kernel_mn_norm[m, n]
                self.info_manifold[m, n] = temp
                self.label_mesh[m, n] = np.array([label_mesh0[m], label_mesh1[n]])

        self.info_manifold = self.info_manifold.reshape( (-1, feamap.shape[1]) )
        self.label_mesh = self.label_mesh.reshape( (-1, 2) )

        # fit the Sigma

        return self.label_mesh.copy(), self.info_manifold.copy()

class Geo_analyzer():
    def __init__(self):
        pass

    def load_data(self, feamap, label):
        '''
        feamap (dict): {'X': [n_observation, n_features], 'R0': [n_observation, n_features], ...}
        label (array [n_observation, n_labels])
        '''
        self.feamap = feamap
        self.label = label
        self.num_label = label.shape[1]

        # create group
        self.ana_group = {}

        for key in feamap:
            self.ana_group[key] = []

            for i in range(self.num_label):
                self.ana_group[key].append( Single_geo_analyzer() )

    def label_dis(self, label_id=None):
        '''show the histogram of label distribution'''
        if label_id is None:
            sns.displot(self.label)
        else:
            sns.displot(self.label[:, label_id])
        plt.show()

    def fit_info_manifold_all(self, label_mesh, label_id=0, kernel_name='gaussian', kernel_width=0.1):
        '''
        fit the info_manifold for all keys but single label
        label_id (int): the ith label
        '''
        for key in self.ana_group:
            self.ana_group[key][label_id].fit_info_manifold(label_mesh, self.feamap[key], self.label[:, label_id], kernel_name=kernel_name, kernel_width=kernel_width)

    def fit_manifold_subspace_all(self, explained_var_thre, label_id=0):
        for key in self.ana_group:
            self.ana_group[key][label_id].fit_manifold_subspace(explained_var_thre)

    def manifold_decoder_score_all(self, feamap_test, label_test, label_id=0):
        score = {}
        for key in self.ana_group:
            _, score[key] = self.ana_group[key][label_id].manifold_decoder_score(feamap_test[key], label_test[:, label_id])
        return score

    def linear_regression_score_all(self, explained_var_thre, feamap_test, label_test, label_id=0):
        score = {}
        for key in self.ana_group:
            _, score[key] = self.ana_group[key][label_id].linear_regression_score(explained_var_thre, self.feamap[key], self.label[:, label_id], feamap_test[key], label_test[:, label_id])
        return score

    def dim_all(self, explained_var_thre, label_id):
        dim = {} # dimensionality of the manifold
        for key in self.ana_group:
            _, dim[key] = self.ana_group[key][label_id].fit_manifold_subspace(explained_var_thre)
        return dim

    def mutual_info_all(self, feamap_test, label_test, label_id=0, sigma=1, normalized=True):
        mi = {} # dimensionality of the manifold
        for key in self.ana_group:
            mi[key] = self.ana_group[key][label_id].mutual_info_score(feamap_test[key], label_test[:, label_id], sigma=sigma, normalized=normalized)
        return mi

    def subspace_var_ratio(self, feamap_test, label_id_proj0, label_id_proj1):
        '''
        calculate the overlapping of two manifold subspace. Testing data will be firstly projected to subspace for label_id_proj0, record the variance var0. Then continue projecting data to the subspace of label_id_proj1, calculate the variance var01. The overlapping variance ratio is var01 / var0. Orthogonal then this number would be zero.
        '''
        var_ratio = {}
        for key in self.ana_group:
            feamap_proj0 = self.ana_group[key][label_id_proj0].pca.transform(feamap_test[key])
            feamap_proj0 = self.ana_group[key][label_id_proj0].pca.inverse_transform(feamap_proj0)
            var0 = vector_variance(feamap_proj0)

            feamap_proj01 = self.ana_group[key][label_id_proj1].pca.transform(feamap_proj0)
            feamap_proj01 = self.ana_group[key][label_id_proj1].pca.inverse_transform(feamap_proj01)
            var01 = vector_variance(feamap_proj01)
            var_ratio[key] = var01 / var0
        return var_ratio

def vector_variance(x):
    '''
    x (array [n_example, n_feature])
        1. calculate the averaged vector across different examples.
        2. sum over the variance of different components
    '''
    var = np.var(x, axis=0)
    var = np.sum(var)
    return var
