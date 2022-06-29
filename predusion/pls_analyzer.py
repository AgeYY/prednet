import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.model_selection import train_test_split
from predusion.geo_tool import unit_vector, angle_between

# pls angle
# pls visualization

class PLS_pair():
    def __init__(self, n_components=1):
        self.pls0 = PLSRegression(n_components=1)
        self.pls1 = PLSRegression(n_components=1)

    def fit(self, neural_x, label, test_size=0.3):
        '''
        neural_x ( [n_sample, n_features] )
        label ([n_sample, n_labels = 2]):
        '''
        self.neural_x_train, self.neural_x_test, self.label_train, self.label_test = train_test_split(neural_x, label, test_size=test_size)

        self.pls0.fit(self.neural_x_train, self.label_train[:, [0]])
        self.pls1.fit(self.neural_x_train, self.label_train[:, [1]])

        self.pls_ax_0 = unit_vector(self.pls0.x_weights_[:, 0])
        self.pls_ax_1 = unit_vector(self.pls1.x_weights_[:, 0])

    def transform(self, data):
        '''
        projecting data to the plane of pls0.x_weights_[:, 0] and pls1.x_weights_[:, 1], and shift the mean of data to center
        data ( [n_sample, n_features] )
        '''

        # find the direction orthogonal to pls_ax_0
        pls_ax_0_orth = unit_vector( self.pls_ax_1 - (self.pls_ax_0 * np.dot(self.pls_ax_0, self.pls_ax_1)) )

        # projecting the data
        data_proj = np.empty( (data.shape[0], 2) )
        data_proj[:, 0] = np.dot(data, self.pls_ax_0)
        data_proj[:, 1] = np.dot(data, pls_ax_0_orth)

        data_proj = data_proj - np.mean(data_proj, axis=0)

        return data_proj

    def fit_transform(self, data):
        return self.transform( self.fit(data) )

    def angle(self):
        ''' Please fit first'''

        return angle_between(self.pls_ax_0, self.pls_ax_1)

    def score(self):
        ''' please fit first '''

        score_lt0 = self.pls0.score(self.neural_x_test, self.label_test[:, [0]])
        score_lt1 = self.pls1.score(self.neural_x_test, self.label_test[:, [1]])

        return score_lt0, score_lt1

    def weight(self):
        ''' please fit first '''
        return self.pls_ax_0, self.pls_ax_1
