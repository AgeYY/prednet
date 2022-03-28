# generate a artificial video based one natural video with predefined curvature
# Read in the natural video
# build up the loss function
# a curvature measurement function + distance to the natural video + constraint the norm of each time (+ a circle mask) point. The first one is the most important
from predusion.video_straight_reader import VS_reader
from predusion.immaker import Batch_gen
from predusion.curvature import Curvature

########## Load the video
imshape = (128, 160)
video_type = 'natural'
video_cate = '01'
scale = '1x'
n_component_video = 5 # the curvature is calculated after dimension reduction to n_component_video

vsread = VS_reader()
video = vsread.read_video(video_type=video_type, video_cate=video_cate, scale=scale) # [number of images in a seq, imshape[0], imshape[1]]

# Calculate the curvature of current natural video
video_flat = video.reshape(1, video.shape[0], -1) # ([n_video, n_video_frames, n_neurons])
n_frame = video.shape[0]
cv = Curvature()
cv.load_data(video_flat)
tg_curvature = cv.curvature_traj(n_component=n_component_video)

def loss_curv(video_vec, n_frame, tg_curvature):
    '''
    video_vec (n_frames * n_pixels in each frame): only one video
    '''
    video_frame = video_vec.reshape(1, n_frame, -1)
    cv.load_data(video_frame)
    curvature = cv.curvature_traj(n_component=n_component_video)
    return (curvature - tg_curvature)**2

art_video_type = 'synthetic'
art_video_cate = '06'
video = vsread.read_video(video_type=art_video_type, video_cate=art_video_cate, scale=scale) # [number of images in a seq, imshape[0], imshape[1]]
video_vec = video.flatten() # flat to an array

dcurv = loss_curv(video_vec, n_frame, tg_curvature)

print(dcurv)
