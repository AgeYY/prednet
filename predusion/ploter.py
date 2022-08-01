# functions for plotting different figures
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from predusion.pls_analyzer import PLS_pair

class Ploter():

    @staticmethod
    def plot_seq_prediction(stimuli, prediction):
        '''
        plot prediction of one sequence
        input:
          stimuli (n_image, *imshape, 3): rgb color
          prediction (n_image, *imshape, 3): the output of Agent() while the output_mode is prediction. The value should be 0 to 255 int
        output:
          fig, ax
        '''

        n_image = stimuli.shape[0]
        fig = plt.figure(figsize = (n_image, 2))
        gs = gridspec.GridSpec(2, n_image)
        gs.update(wspace=0., hspace=0.)

        for t, sq_s, sq_p in zip(range(n_image), stimuli, prediction):
            plt.subplot(gs[t])

            ## the image can be ploted without explicit normalization anyway
            #sq_s_norm = cv2.normalize(sq_s, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            #sq_p_norm = cv2.normalize(sq_p, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            #sq_s_norm = sq_s_norm.astype(np.uint8)
            #sq_p_norm = sq_p_norm.astype(np.uint8)

            sq_s_norm = sq_s
            sq_p_norm = sq_p

            plt.imshow(sq_s_norm)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

            plt.subplot(gs[t + n_image])
            plt.imshow(sq_p_norm)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        return fig, gs

    def plot_seq(seq):
        '''
        plot a sequence
        input:
          seq (n_image, *imshape, 3): rgb color
        output:
          fig, ax
        '''

        n_image = seq.shape[0]
        fig = plt.figure(figsize = (n_image, 1))
        gs = gridspec.GridSpec(1, n_image)
        gs.update(wspace=0., hspace=0.)

        for t, sq in zip(range(n_image), seq):
            plt.subplot(gs[t])
            plt.imshow(sq.astype(np.uint8))
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        return fig, gs

class Ploter_dim_reduction():
    def __init__(self, method='mds', n_neighbors=2, n_components=3):
        self.method = method
        if method=='mds':
            self.embedding = MDS(n_components=n_components)
        elif method=='lle':
            self.embedding = LocallyLinearEmbedding(n_components=n_components)
        elif method=='isomap':
            self.embedding = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        elif method=='pca':
            self.embedding = PCA(n_components=n_components)
        elif method=='pls_pair':
            self.embedding = PLS_pair(n_components=1) # two latent variables, pick the first component for each latent variable

    def fit(self, data, label=None):
        '''
        data ([sample, feature])
        n_component (int): 2 or 3 dimension visualization
        label ([sample, n_label]): required when the method is pls
        '''
        if self.method == 'pls_pair':
            return self.embedding.fit(data, label)
        else:
            return self.embedding.fit(data)

    def transform(self, data):
        '''
        data ([sample, feature])
        '''
        return self.embedding.transform(data)

    def fit_transform(self, data, fit_label):
        self.fit(data, fit_label)
        return self.transform(data)

    @staticmethod
    def plot_helper(ax, plot_func, mode, *para, **kwargs):

        name = plot_func + mode
        if (name == 'plot2D') or (name == 'plot3D'):
            return ax.plot(*para, **kwargs)
        elif name == 'scatter2D':
            return ax.scatter(*para, **kwargs)
        elif name == 'scatter3D':
            return ax.scatter3D(*para, **kwargs)

    def plot_dimension_reduction(self, data, plot_func='scatter', fit_label=None, colorinfo=None, title=None, save_fig=None, ax=None, fig=None, cax=None, fit=False, mode='2D', alpha=1, marker='o'):
        '''
        data ( [n_sample, n_feature])
        label ( [n_sample, 2] ): only 2 labels for pls_pair
        save_fig (str): the path for saving figure
        fit (bool): refit the data, or use the fitted embedding
        mode (str): 2D or 3D
        '''
        if fit:
            data_trans = self.fit_transform(data, fit_label)
        else:
            data_trans = self.transform(data)

        if fig is None:
            fig = plt.figure()

        if mode == '2D':
            if ax is None:
                ax = fig.add_subplot()

            if cax is None:
                cax = fig.add_axes([0.27, 0.8, 0.5, 0.05]) # colorbar

            if not (colorinfo is None):
                im = self.plot_helper(ax, plot_func, mode, data_trans[:, 0], data_trans[:, 1], c=colorinfo.flatten(), cmap="viridis", alpha=alpha, marker=marker)
                fig.colorbar(im, cax=cax, orientation = 'horizontal')
            else:
                im = self.plot_helper(ax, plot_func, mode, data_trans[:, 0], data_trans[:, 1], alpha=alpha, marker=marker)

        elif mode == '3D':
            if ax is None:
                ax = plt.axes(projection='3d')

            if cax is None:
                cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

            if not (colorinfo is None):
                im = self.plot_helper(ax, plot_func, mode, data_trans[:, 0], data_trans[:, 1], data_trans[:, 2], c=colorinfo.flatten(), cmap = "viridis", depthshade=False, alpha=alpha, marker=marker)
                fig.colorbar(im, cax = cax, orientation = 'horizontal')
            else:
                im = self.plot_helper(ax, plot_func, mode, data_trans[:, 0], data_trans[:, 1], data_trans[:, 2], depthshade=False, alpha=alpha, marker=marker)

        ax.set_title(title)
        if not (save_fig is None):
            fig.savefig(save_fig)
        return fig, ax

    def add_line(self, line, ax, mode='2D', **kwargs):
        '''
        add a line to ax after plot_dimension_reduction. mode must be the same as plot_dim_reduction
        line ([2, n_features]): the first raw is the initial point, second row is the final one
        '''
        line_trans = self.embedding.transform(line)
        line_trans = line_trans - line_trans[0, :] # shift to the origin of the PC
        #xlow, xhigh = ax.get_xlim()
        #tangent = (line_trans[1, 1] - line_trans[1, 0]) / (line_trans[0, 1] - line_trans[0, 0])
        #line_trans[0, 0], line_trans[1, 0] = xlow, xhigh
        #line_trans[0, 1], line_trans[1, 1] = xlow * tangent, xhigh * tangent

        if mode == '2D':
            ax.plot(line_trans[:, 0], line_trans[:, 1], **kwargs)
        elif mode == '3D':
            ax.plot(line_trans[:, 0], line_trans[:, 1], line_trans[:, 2], **kwargs)
        return ax

def align_data(data, delta):
    '''
    align the mean of different information manifolds to a line
    '''
    if delta is None:
        return data

    line = np.zeros(data.shape[1:])
    line[:, 0] = delta * np.arange(data.shape[1])
    data_mean = np.mean(data, axis=0)
    delta_data = data_mean - line
    shift_neural_x = np.tile(np.expand_dims(delta_data, axis=0), (12, 1 , 1))
    return data - shift_neural_x

def plot_dimension_reduction(data, colorinfo=None, method='mds', n_components=2, title='', n_neighbors=8, align_delta=None, save_fig=True, ax=None, fig=None, cax=None):
    '''
    data ([sample, feature])
    n_component (int): 2 or 3 dimension visualization
    '''
    data = align_data(data, align_delta)
    if method=='mds':
        embedding = MDS(n_components=n_components)
    elif method=='lle':
        embedding = LocallyLinearEmbedding(n_components=n_components)
    elif method=='isomap':
        embedding = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    elif method=='pca':
        embedding = PCA(n_components=n_components)

    data_transformed = embedding.fit_transform(data.reshape([-1, data.shape[-1]]))

    if fig is None:
        fig = plt.figure()

    if n_components == 2:
        if ax is None:
            ax = fig.add_axes()

        if cax is None:
            cax = fig.add_axes([0.27, 0.8, 0.5, 0.05]) # colorbar

        if not (colorinfo is None):
            im = ax.scatter(data_transformed[:, 0], data_transformed[:, 1], c=colorinfo.flatten(), cmap="viridis")
            fig.colorbar(im, cax=cax, orientation = 'horizontal')
        else:
            im = ax.scatter(data_transformed[:, 0], data_transformed[:, 1])

    elif n_components == 3:
        if ax is None:
            ax = plt.axes(projection='3d')

        if cax is None:
            cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

        if not (colorinfo is None):
            im = ax.scatter3D(data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2], c=colorinfo.flatten(), cmap = "viridis", depthshade=False)
            fig.colorbar(im, cax = cax, orientation = 'horizontal')
        else:
            im = ax.scatter3D(data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2], depthshade=False)

    ax.set_title(title)
    if save_fig:
        fig.savefig('./figs/' + title + '.pdf')
    return fig, ax

def plot_layer_error_bar_helper(score, n_layer, layer_order, ax, error_bar_method='std'):
    '''
    score (dict): for example, score = {'X': [0.1, 0.2], 'R0': [0.5, 0.3]} where each list contains repeated results
    n_layer (int): number of layers, should be equal to the number of keys
    layer_order (list of str): ['X', 'R0', 'R1', ...]
    '''
    score_order = {lo: score[lo] for lo in layer_order}

    score_error = np.zeros(n_layer)
    score_mean = np.zeros(n_layer)

    i = 0
    for key in score_order:
        score_mean[i] = np.mean(score_order[key])
        if error_bar_method=='std':
            score_error[i] = np.std(score_order[key])
        else: # sem
            score_error[i] = np.std(score_order[key]) / np.sqrt(len(score_order[key]))
        i += 1

    ax.scatter(np.arange(-1, n_layer - 1), score_mean)
    ax.errorbar(np.arange(-1, n_layer - 1), score_mean, yerr=score_error)
    ax.plot(np.arange(-1, n_layer - 1), score_mean)
    return ax

if __name__ == '__main__':
    import os
    import predusion.immaker as immaker
    from predusion.immaker import Seq_gen
    from kitti_settings import *
    from predusion.agent import Agent

    ##### load the prednet
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

    sub = Agent()
    sub.read_from_json(json_file, weights_file)

    ###### generate images
    imshape = (128, 160)
    square = immaker.Square(imshape)

    im = square.set_full_square(color=[0, 0, 100])
    seq_repeat = Seq_gen().repeat(im, 5)[None, ...] # add a new axis to show there's only one squence

    ##### prediction
    seq_pred = sub.output(seq_repeat)
    fig, gs = Ploter().plot_seq_prediction(seq_repeat[0], seq_pred[0])
    plt.show()
