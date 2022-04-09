# generate a artificial video based one natural video with predefined curvature
from scipy.optimize import minimize as sci_minimize
from sklearn.decomposition import PCA
import numpy as np
from predusion.curvature import Curvature

class Artf_video():
    '''
    generate artificial videos with mean curvature equal to the target curvature.
    '''
    def __init__(self, n_component=5, alpha=1, fix_end=True):
        '''
        fix_end (bool): we provide two constraints.
          1. fix_end = True. the first and final video frame, after dimensiona reduction, must be identical.
          2. fix_end = False. The first frame is identical, the norm (frame i - fram_{i-1}) are identical, but the final frame not necessarily identical
        '''
        self.cv = Curvature()
        self.pca = PCA(n_components=n_component)
        self.n_component = n_component
        self.alpha=alpha # the hyperparameter in contronling the norm = 1 in loss function
        self.nat_video = None
        self.fix_end=fix_end # fix the first and final frame to be the same as artificial.

    def set_n_component(self, n_component):
        '''
        set the number of principal compoennts
        '''
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
        self.n_frame = nat_video.shape[0]
        self.imshape = (nat_video.shape[1:])
        # do the pca
        nat_video_pca = self.pca.fit_transform(nat_video.reshape(self.n_frame, -1))
        # store x0
        self.x0_pca = nat_video_pca[0]
        # store final frame
        self.xf_pca = nat_video_pca[-1]
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

        if self.fix_end:
            # require the first and final frame to be the same as the natural video, after pca
            y_f = np.sum(tildew_mat, axis=0) + self.x0_pca
            loss2 = np.linalg.norm(y_f - self.xf_pca)**2
        else:
            loss2=0

        return loss1 + loss2

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

        return art_video, result

    def recons_video(self, tildew_mat):
        '''
        reconstruct the video based on theta
        input:
          init_frame (str): natural -- the same as self.x0; random -- uniform from [0, 1] in every principal component. This parameter would be natural if fix_end is true
          tildew_mat (array, float, [self.n_frame - 1, self.n_component])
        output:
          art_video (array, float, same shape as the natural image)
        '''
        video_pca = np.zeros( (self.n_frame, self.n_component) )

        video_pca[0] = self.x0_pca

        for i in range(tildew_mat.shape[0]): # from the fist frame to the last one
            if self.fix_end:
                video_pca[i + 1] = video_pca[i] + tildew_mat[i] # the shape is 
            else:
                video_pca[i + 1] = video_pca[i] + self.vec_norm_pca[i] * tildew_mat[i] / np.linalg.norm(tildew_mat[i]) # the shape is 
        # convert theta to the artificial video
        video = np.zeros(( self.n_frame, *self.imshape ))
        for i, frame in enumerate(video_pca):
            video[i] = self.pca.inverse_transform( frame ).reshape(self.imshape)
        return video

from kitti_settings import *
from predusion.video_straight_reader import VS_reader
from predusion.curvature import Curvature

def gen_artf_video_from_vs_reader(imshape=(128, 160), n_component_video=5, fix_end=True, n_video=2, verbose=False):
    '''
    generate artificial videos from natural videos in VS reader
    input:
      n_video (int): the number of artificial videos, cannot be larger than the number of natural videos which is 11. Larger then the output number of video would be 11
      verbose (bool): print out the curvature values
    output:
      artf_video_batch (n_video, n_frame, *imshape)
    '''
    video_type = 'natural'
    scale = '1x'

    ########## Load the natural video and find the target curvature
    vsread = VS_reader()
    nat_video_all = vsread.read_video_all_ppd(video_type=video_type, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]
    
    nat_video_flat_all = nat_video_all.reshape(nat_video_all.shape[0], nat_video_all.shape[1], -1) # ([n_video, n_video_frames, n_neurons])
    n_frame = nat_video_all.shape[1]

    
    artf_video_batch = [] # storing all artificial videos
    
    for i, nat_video_flat_i in enumerate(nat_video_flat_all):
        if not (i < n_video): break # stop when video are enough

        nat_video_flat = nat_video_flat_i[None, ...]
        nat_video = nat_video_all[i]
        
        cv = Curvature()
        cv.load_data(nat_video_flat)
        tg_curvature = cv.curvature_traj(n_component=n_component_video)
            
        ########## Compute the loss function
        artf_gen = Artf_video(n_component=5, alpha=0.1, fix_end=fix_end)
        artf_gen.load_natural_video(nat_video)
        
        artf_video, result = artf_gen.minimize(tg_curvature)

        artf_video_batch.append(artf_video)

        if verbose:
            cv.load_data(artf_video.reshape((1, nat_video.shape[0], -1)))
            artf_curv = cv.curvature_traj(n_component=5)

            print('curvature of natural video: {}'.format(tg_curvature))
            print('curvature of your artificial video: {}'.format(artf_curv))


    return np.array(artf_video_batch)





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from predusion.video_straight_reader import VS_reader
    from predusion.curvature import Curvature

    imshape = (128, 160)
    video_type = 'natural'
    video_cate = '07'
    scale = '1x'
    n_component_video = 5 # the curvature is calculated after dimension reduction to n_component_video
    tg_curv_mannual = 0.961
    fix_end=True
    
    ########## Load the natural video and find the target curvature
    vsread = VS_reader()
    nat_video = vsread.read_video_ppd(video_type=video_type, video_cate=video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]
    
    nat_video_flat = nat_video.reshape(1, nat_video.shape[0], -1) # ([n_video, n_video_frames, n_neurons])
    n_frame = nat_video.shape[0]
    cv = Curvature()
    cv.load_data(nat_video_flat)
    print('nat:', cv.curvature_traj(n_component=n_component_video))
    if tg_curv_mannual is None:
        tg_curvature = cv.curvature_traj(n_component=n_component_video)
    else:
        tg_curvature = tg_curv_mannual
        
        ########## Compute the loss function
        artf_gen = Artf_video(n_component=5, alpha=0.1, fix_end=fix_end)
        artf_gen.load_natural_video(nat_video)
        
        artf_video, result = artf_gen.minimize(tg_curvature)
        
        cv.load_data(artf_video.reshape((1, nat_video.shape[0], -1)))
        artf_curv = cv.curvature_traj(n_component=5)
        print('curvature of your artificial video: {}'.format(artf_curv))
        print('target curvature: {}'.format(tg_curvature))
        
        ######### Show the video
        for im in artf_video:
            plt.imshow(im)
            plt.show()