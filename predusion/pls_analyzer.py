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
        label ([n_sample, n_labels = 2]):
        '''
        self.neural_x_train, self.neural_x_test, self.label_train, self.label_test = train_test_split(neural_x, label, test_size=test_size)

        self.pls0.fit(self.neural_x_train, self.label_train[:, [0]])
        self.pls1.fit(self.neural_x_train, self.label_train[:, [1]])

    #score_lt0 = pls.score(neural_x_test, label_test[:, [0]])
    #score_lt1 = pls.score(neural_x_test, label_test[:, [1]])

    def angle(self):
        ''' Please fit first'''

        pls_ax_0 = self.pls0.x_weights_[:, 0]
        pls_ax_1 = self.pls1.x_weights_[:, 0]

        return angle_between(pls_ax_0, pls_ax_1)

    def score(self):
        ''' please fit first '''

        score_lt0 = self.pls0.score(self.neural_x_test, self.label_test[:, [0]])
        score_lt1 = self.pls1.score(self.neural_x_test, self.label_test[:, [1]])

        return score_lt0, score_lt1

    def weight(self):
        ''' please fit first '''
        return pls_ax_0, pls_ax_1
