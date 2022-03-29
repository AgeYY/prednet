# generate a artificial video based one natural video with predefined curvature
# Read in the natural video
# build up the loss function
# a curvature measurement function + distance to the natural video + constraint the norm of each time (+ a circle mask) point. The first one is the most important
from scipy.optimize import minimize as sci_minimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.curvature import Curvature

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

    def loss_curv(self, theta, tg_curvature):
        '''
        loss function: (sum{ cos(\theta_i) \dot cos(\theta_i+1) }/(n - 2) - cos(tg_curvature))^2 + alpha * sum{ norm{cos(\theta_i)} - 1}^2 / (n-1)
        input:
          theta (array, float, [(self.n_frame - 1) * self.n_component])
        output:
          loss (float)
        '''
        theta_mat = theta.reshape(-1, self.n_component) # [(n_frame - 1), n_compoent]
        cos_theta = np.cos(theta_mat)
        cos_dot = 0
        for i in range(theta_mat.shape[0]-1):
            #cos_dot += np.dot( cos_theta[i], cos_theta[i + 1] )
            cos_dot += (np.dot( cos_theta[i], cos_theta[i + 1] ) - np.cos(tg_curvature))**2
        #cos_dot = cos_dot / (theta_mat.shape[0]-2)
        #loss1 = (cos_dot - np.cos(tg_curvature))**2
        loss1 = cos_dot / (theta_mat.shape[0]-2)

        loss2 = 0
        for i in range(theta_mat.shape[0]):
            loss2 += (np.linalg.norm(cos_theta[i]) - 1)**2
        loss2 = loss2 / theta_mat.shape[0]

        print(loss1 + self.alpha * loss2)
        return loss1 + self.alpha * loss2

    def minimize(self, tg_curvature, options=None):
        '''
        minimize the loss function given video_init and target curvature
        input:
          tg_curvature (float): the target curvature
          random_seed (int)
        output:
        '''
        # initialize theta, as an flatten array. shape? Let's say shape of natural video is (n_frame - 2) * reduced dimension (after PCA)
        theta = np.random.uniform(low=0, high=2 * np.pi, size=(self.n_frame - 1) * self.n_component)
        # feed theta into the minimize function
        result = sci_minimize(self.loss_curv, theta, args=(tg_curvature), options=options)
        # get the result theta from result
        theta_mat = result.x.reshape((n_frame-1, self.n_component))
        # reconstruct the reduced video
        art_video = self.recons_video(theta_mat)
        # convert the dimension
        return art_video, result

    def recons_video(self, theta_mat):
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
        for i in range(theta_mat.shape[0]): # from the fist frame to the last one
            video_pca[i + 1] = video_pca[i] + self.vec_norm_pca[i] * theta_mat[i] # the shape is 
        # convert theta to the artificial video
        video = np.zeros(( self.n_frame, *self.imshape ))
        for i, frame in enumerate(video_pca):
            video[i] = self.pca.inverse_transform( frame ).reshape(self.imshape)
        return video

imshape = (128, 160)
video_type = 'natural'
video_cate = '01'
scale = '1x'
n_component_video = 5 # the curvature is calculated after dimension reduction to n_component_video

########## Load the natural video and find the target curvature
vsread = VS_reader()
nat_video = vsread.read_video_ppd(video_type=video_type, video_cate=video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]

nat_video_flat = nat_video.reshape(1, nat_video.shape[0], -1) # ([n_video, n_video_frames, n_neurons])
n_frame = nat_video.shape[0]
cv = Curvature()
cv.load_data(nat_video_flat)
tg_curvature = cv.curvature_traj(n_component=n_component_video)

########## Load the artificial video as an initialization
art_video_type = 'synthetic'
video = vsread.read_video_ppd(video_type=art_video_type, video_cate=video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]

########## Compute the loss function
artf_gen = Artf_video(n_component=5, alpha=0.1)
artf_gen.load_natural_video(nat_video)

artf_video, result = artf_gen.minimize(tg_curvature)

cv.load_data(artf_video.reshape((1, nat_video.shape[0], -1)))
artf_curv = cv.curvature_traj(n_component=n_component_video)
print(artf_curv)
print(tg_curvature)
print(result.fun)

########## Show the video
for im in video:
    plt.imshow(im)
    plt.show()
