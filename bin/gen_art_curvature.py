# generate a artificial video based one natural video with predefined curvature
# Read in the natural video
# build up the loss function
# a curvature measurement function + distance to the natural video + constraint the norm of each time (+ a circle mask) point. The first one is the most important
from scipy.optimize import minimize
import numpy as np

from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.curvature import Curvature

class Artf_video():
    '''
    generate artificial videos with mean curvature equal to the target curvature
    '''
    def __init__(self):
        self.cv = Curvature()
    def load_natural_video():
        pass
    def loss_curv(self, video_vec, n_frame, tg_curvature):
        '''
        loss function
        video_vec (ndarray [n_frame * n_pixels in each frame])
        '''
        video_frame = video_vec.reshape(1, n_frame, -1)
        self.cv.load_data(video_frame)
        curvature = self.cv.curvature_traj(n_component=n_component_video)
        return (np.cos(curvature) - np.cos(tg_curvature))**2

    def minimize(self, video_init, tg_curvature):
        '''
        minimize the loss function given video_init and target curvature
        video_init (ndarray [1, n_frame, imshape[0], imshape[1]]): initial guess
        tg_curvature (float): the target curvature
        '''
        n_frame = video_init.shape[1]
        imshape = (video_init.shape[2:4]) # shape of a frame
        result = minimize(self.loss_curv, video_init.flatten(), args=(n_frame, tg_curvature), options={'maxiter': 2})
        video = result.x.reshape(1, n_frame, imshape[0], imshape[1])
        return video, result

    def _process_result():
        '''
        process the result of minimize back to video with shape [1, n_frame, n_pixels in each frame]
        '''
        pass

#class Artif_video_natural(Artf_video):
#    '''
#    generate artificial videos with mean curvature equal to the target curvature. The norm of the difference between two adjacent video frames are the same. This can also avoid two frames non-identical, so that curvature can be normally computed
#    '''
#    def load_natural_video(self, video):
#        '''
#        load one natural video with shape [1, n_frame, imshape[0], imshape[1]]
#        '''
#        self.frame_init = video[0, 0] # the initial video frame
#        self.vec_norm = np.norm(
#            np.diff( video.reshape((n_frame, -1)), axis=0)\,
#            axis=1
#        )
#        print(vec_norm.shape)
#        pass
#
#    def theta2video():
#        '''
#        convert theta to videos
#        '''
#        pass
#    def loss_curv(self, video_vec, n_frame, tg_curvature):
#        pass

imshape = (3, 3)
video_type = 'natural'
video_cate = '01'
scale = '1x'
n_component_video = 5 # the curvature is calculated after dimension reduction to n_component_video

########## Load the natural video and find the target curvature
vsread = VS_reader()
video = vsread.read_video_ppd(video_type=video_type, video_cate=video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]

video_flat = video.reshape(1, video.shape[0], -1) # ([n_video, n_video_frames, n_neurons])
n_frame = video.shape[0]
cv = Curvature()
cv.load_data(video_flat)
tg_curvature = cv.curvature_traj(n_component=n_component_video)

########## Load the artificial video as an initialization
art_video_type = 'synthetic'
art_video_cate = '01'
video = vsread.read_video_ppd(video_type=art_video_type, video_cate=art_video_cate, scale=scale, imshape=imshape) # [number of images in a seq, imshape[0], imshape[1]]

########## Compute the loss function
#artf_gen = Artf_video()
#video, result = artf_gen.minimize(video[None, :], tg_curvature)
#print(result.success, result.fun)

