# generate a artificial video based one natural video with predefined curvature
# Read in the natural video
# build up the loss function
# a curvature measurement function + distance to the natural video + constraint the norm of each time (+ a circle mask) point. The first one is the most important
from scipy.optimize import minimize as sci_minimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

from predusion.immaker import Batch_gen
from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.curvature import Curvature
from predusion.agent import Agent

from kitti_settings import *

#class Artf_video():
#    '''
#    generate artificial videos with mean curvature equal to the target curvature. The norm of the difference between two adjacent video frames are the same. This can also avoid two frames non-identical, so that curvature can be normally computed
#    '''
#    def __init__(self, n_component=5, alpha=1):
#        self.cv = Curvature()
#        self.pca = PCA(n_components=n_component)
#        self.n_component = n_component
#        self.alpha=alpha # the hyperparameter in contronling the norm = 1 in loss function
#        self.nat_video = None
#
#    def set_n_component(self, n_component):
#        self.n_component=n_component
#        self.pca = PCA(n_components=n_component)
#
#        if not (self.nat_video is None):
#            self.load_natural_video(self.nat_video)
#
#    def load_natural_video(self, nat_video):
#        '''
#        input:
#          nat_video ([n_frame, imshape[0], imshape[1]])
#        stored data:
#          self.norm_pca (array, float, [n_frame - 1]): norm of x_(i+1)_pca - x_i_pca
#        '''
#        # store the video
#        self.nat_video = nat_video
#        # store parameters: n_frame, imshape
#        self.n_frame = video.shape[0]
#        self.imshape = (video.shape[1:])
#        # do the pca
#        nat_video_pca = self.pca.fit_transform(nat_video.reshape(self.n_frame, -1))
#        # store x0
#        self.x0_pca = nat_video_pca[0]
#        # store the norm
#        self.vec_norm_pca = np.linalg.norm(
#            np.diff( nat_video_pca, axis=0), \
#            axis=1
#        )
#
#    def curv_natural_video():
#        pass
#
#    def loss_curv(self, theta, tg_curvature):
#        '''
#        loss function: (sum{ cos(\theta_i) \dot cos(\theta_i+1) }/(n - 2) - cos(tg_curvature))^2 + alpha * sum{ norm{cos(\theta_i)} - 1}^2 / (n-1)
#        input:
#          theta (array, float, [(self.n_frame - 1) * self.n_component])
#        output:
#          loss (float)
#        '''
#        theta_mat = theta.reshape(-1, self.n_component) # [(n_frame - 1), n_compoent]
#        cos_theta = np.cos(theta_mat)
#        cos_dot = 0
#        for i in range(theta_mat.shape[0]-1):
#            cos_dot += np.dot( cos_theta[i], cos_theta[i + 1] )
#        cos_dot = cos_dot / (theta_mat.shape[0]-2)
#        loss1 = (cos_dot - np.cos(tg_curvature))**2
#
#        loss2 = 0
#        for i in range(theta_mat.shape[0]):
#            loss2 += (np.linalg.norm(cos_theta[i]) - 1)**2
#        loss2 = loss2 / theta_mat.shape[0]
#
#        return loss1 + self.alpha * loss2
#
#    def minimize(self, tg_curvature, options=None):
#        '''
#        minimize the loss function given video_init and target curvature
#        input:
#          tg_curvature (float): the target curvature
#          random_seed (int)
#        output:
#        '''
#        # initialize theta, as an flatten array. shape? Let's say shape of natural video is (n_frame - 2) * reduced dimension (after PCA)
#        theta = np.random.uniform(low=0, high=2 * np.pi, size=(self.n_frame - 1) * self.n_component)
#        # feed theta into the minimize function
#        result = sci_minimize(self.loss_curv, theta, args=(tg_curvature), options=options)
#        # get the result theta from result
#        theta_mat = result.x.reshape((n_frame-1, self.n_component))
#        # reconstruct the reduced video
#        art_video = self.recons_video(theta_mat)
#        # convert the dimension
#        return art_video, result
#
#    def recons_video(self, theta_mat):
#        '''
#        reconstruct the video based on theta
#        input:
#          theta_mat (array, float, [self.n_frame - 1, self.n_component])
#        output:
#          art_video (array, float, same shape as the natural image)
#        '''
#        video_pca = np.zeros( (self.n_frame, self.n_component) )
#        #video_pca[0] = self.x0_pca
#        video_pca[0] = np.random.uniform(low=0, high=1, size=(self.x0_pca.shape))
#        for i in range(theta_mat.shape[0]): # from the fist frame to the last one
#            video_pca[i + 1] = video_pca[i] + self.vec_norm_pca[i] * np.cos(theta_mat[i]) # the shape is 
#        # convert theta to the artificial video
#        video = np.zeros(( self.n_frame, *self.imshape ))
#        for i, frame in enumerate(video_pca):
#            video[i] = self.pca.inverse_transform( frame ).reshape(self.imshape)
#        return video

class Artf_video():
    '''
    generate artificial videos with mean curvature equal to the target curvature. The norm of the difference between two adjacent video frames are the same. This can also avoid two frames non-identical, so that curvature can be normally computed
    '''
    def __init__(self, n_component=5, alpha=1):
        self.cv = Curvature()
        self.pca = PCA(n_components=n_component)
        self.n_component = n_component
        self.alpha=alpha # the hyperparameter in contronling the norm = 1 in loss function
        self.nat_video = None

    def set_n_component(self, n_component):
        self.n_component=n_component
        self.pca = PCA(n_components=n_component)

        if not (self.nat_video is None):
            self.load_natural_video(self.nat_video)

    def load_natural_video(self, nat_video):
        '''
        input:
          nat_video ([n_frame, imshape[0], imshape[1]])
        stored data:
          self.norm_pca (array, float, [n_frame - 1]): norm of x_(i+1)_pca - x_i_pca
        '''
        # store the video
        self.nat_video = nat_video
        # store parameters: n_frame, imshape
        self.n_frame = video.shape[0]
        self.imshape = (video.shape[1:])
        # do the pca
        nat_video_pca = self.pca.fit_transform(nat_video.reshape(self.n_frame, -1))
        # store x0
        self.x0_pca = nat_video_pca[0]
        # store the norm
        self.vec_norm_pca = np.linalg.norm(
            np.diff( nat_video_pca, axis=0), \
            axis=1
        )

    def curv_natural_video():
        pass

    def loss_curv(self, tildew, tg_curvature):
        '''
        loss function: (sum{ cos(\theta_i) \dot cos(\theta_i+1) }/(n - 2) - cos(tg_curvature))^2 + alpha * sum{ norm{cos(\theta_i)} - 1}^2 / (n-1)
        input:
          tildew (array, float, [(self.n_frame - 1) * self.n_component])
        output:
          loss (float)
        '''
        tildew_mat = tildew.reshape(-1, self.n_component) # [(n_frame - 1), n_compoent]
        ang = 0
        for i in range(tildew_mat.shape[0]-1):
            cos_dot = np.dot( tildew_mat[i], tildew_mat[i + 1] ) / np.linalg.norm(tildew_mat[i]) / np.linalg.norm(tildew_mat[i + 1])
            ang += np.arccos(cos_dot)
        ang = ang / (tildew_mat.shape[0]-1)
        loss1 = (ang - tg_curvature)**2

        return loss1

    def minimize(self, tg_curvature, options=None):
        '''
        minimize the loss function given video_init and target curvature
        input:
          tg_curvature (float): the target curvature
          random_seed (int)
        output:
        '''
        # initialize theta, as an flatten array. shape? Let's say shape of natural video is (n_frame - 2) * reduced dimension (after PCA)
        tildew = np.random.uniform(low=0, high=1, size=(self.n_frame - 1) * self.n_component)
        # feed theta into the minimize function
        result = sci_minimize(self.loss_curv, tildew, args=(tg_curvature), options=options)
        # get the result theta from result
        tildew_mat = result.x.reshape((self.n_frame-1, self.n_component))
        # reconstruct the reduced video
        art_video = self.recons_video(tildew_mat)
        # convert the dimension
        return art_video, result

    def recons_video(self, tildew_mat):
        '''
        reconstruct the video based on theta
        input:
          theta_mat (array, float, [self.n_frame - 1, self.n_component])
        output:
          art_video (array, float, same shape as the natural image)
        '''
        video_pca = np.zeros( (self.n_frame, self.n_component) )
        #video_pca[0] = self.x0_pca
        video_pca[0] = np.random.uniform(low=0, high=1, size=(self.x0_pca.shape))
        for i in range(tildew_mat.shape[0]): # from the fist frame to the last one
            video_pca[i + 1] = video_pca[i] + self.vec_norm_pca[i] * tildew_mat[i] / np.linalg.norm(tildew_mat[i]) # the shape is 
        # convert theta to the artificial video
        video = np.zeros(( self.n_frame, *self.imshape ))
        for i, frame in enumerate(video_pca):
            video[i] = self.pca.inverse_transform( frame ).reshape(self.imshape)
        return video

imshape = (128, 160)
video_type = 'natural'
video_cate = '07'
scale = '1x'
n_component_video = 5 # the curvature is calculated after dimension reduction to n_component_video
tg_curv_mannual = 0.5

########## Load the natural video and find the target curvature
vsread = VS_reader()
nat_video = vsread.read_video_ppd(video_type=video_type, video_cate=video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]

nat_video_flat = nat_video.reshape(1, nat_video.shape[0], -1) # ([n_video, n_video_frames, n_neurons])
n_frame = nat_video.shape[0]
cv = Curvature()
cv.load_data(nat_video_flat)
if tg_curv_mannual is None:
    tg_curvature = cv.curvature_traj(n_component=n_component_video)
else:
    tg_curvature = tg_curv_mannual

########## Load the artificial video as an initialization
art_video_type = 'synthetic'
video = vsread.read_video_ppd(video_type=art_video_type, video_cate=video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]

########## Compute the loss function
artf_gen = Artf_video(n_component=5, alpha=0.1)
artf_gen.load_natural_video(nat_video)

artf_video, result = artf_gen.minimize(tg_curvature)

cv.load_data(artf_video.reshape((1, nat_video.shape[0], -1)))
artf_curv = cv.curvature_traj(n_component=5)
print(artf_curv)
print(tg_curvature)
print(result.fun)

######### Show the video
for im in video:
    plt.imshow(im)
    plt.show()

###### load the prednet
#json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
#weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
#n_component = 5
#
#video_ppd = Batch_gen.process_grey_video(artf_video[None, ...], imshape=imshape) # process the video
#
#batch_size = video_ppd.shape[0]
#
#sub = Agent()
#sub.read_from_json(json_file, weights_file)
#
#
#output = sub.output_multiple(video_ppd, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)
#
########### curverature origin
#cv = Curvature()
#
#ct_mean = {}
#ct_mean_pca = {}
#
#for key in output:
#    output[key] = output[key].reshape(output[key].shape[0], output[key].shape[1], -1) # flatten (n_video, n_frames, n_neurons)
#    cv.load_data(output[key])
#
#    ct_mean[key] = cv.curvature_traj(cutoff=cutoff, n_component=None) # ignore the first two video frames due to the pool prediction of the prednet
#    ct_mean_pca[key] = cv.curvature_traj(cutoff=cutoff, n_component=n_component) # ignore the first two video frames due to the pool prediction of the prednet
#
#video_flat = video.reshape((video.shape[0], video.shape[1], -1))
#cv.load_data(video_flat)
#ct_mean_video = cv.curvature_traj(cutoff=cutoff)
#
#ct_mean_video_pca = cv.curvature_traj(cutoff=cutoff, n_component=n_component)
#
########### Change of curvature
#ct_mean_change = {}
#ct_mean_change_pca = {}
#
#ct_mean_change_median = {}
#ct_mean_change_pca_median = {}
#
#ct_mean_change_ci = {}
#ct_mean_change_pca_ci = {}
#
#for key in ct_mean:
#    ct_mean_change[key] = ct_mean[key] - ct_mean_video
#    ct_mean_change_median[key] = np.median(ct_mean_change[key])
#    ct_mean_change_ci[key] = np.array( confidence_interval(ct_mean_change[key], ci_bound=ci_bound, measure=np.median, n_resamples=ci_resample) )
#
#    ct_mean_change_pca[key] = ct_mean_pca[key] - ct_mean_video_pca
#    ct_mean_change_pca_median[key] = np.median(ct_mean_change_pca[key])
#    ct_mean_change_pca_ci[key] = np.array( confidence_interval(ct_mean_change_pca[key], ci_bound=ci_bound, measure=np.median, n_resamples=ci_resample) )
#
########### plot out the median
#def plot_curv_layer(ct_mean_change_median, ct_mean_change_ci):
#    for key in ct_mean_change_median:
#        x, y = [int(key[1])], [ct_mean_change_median[key]]
#        yerr = (ct_mean_change_ci[key] - y)[..., None]
#        yerr = np.abs(yerr)
#        if 'E' in key:
#            plt.scatter(x, y, c='green')
#            plt.errorbar(x, y, yerr=yerr, c='green')
#        if 'R' in key:
#            plt.scatter(x, y, c='red')
#            plt.errorbar(x, y, yerr=yerr, c='red')
#
#plt.figure()
#plot_curv_layer(ct_mean_change_median, ct_mean_change_ci)
#plt.axhline(y = 0, linestyle = '--', linewidth = 1, color = 'black')
#plt.legend()
#plt.xlabel('Module ID')
#plt.ylabel('Change of curvature \n (curvature of neurons - curvature of videos)')
#plt.show()
#
#plt.figure()
#plot_curv_layer(ct_mean_change_pca_median, ct_mean_change_pca_ci)
#plt.axhline(y = 0, linestyle = '--', linewidth = 1, color = 'black')
#plt.legend()
#plt.xlabel('Module ID')
#plt.ylabel('Change of curvature \n (curvature of neurons - curvature of videos)')
#plt.show()
