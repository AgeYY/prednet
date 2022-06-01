# generate moving bar images and dataset as moving_bar_train.hkl.

from predusion.immaker import Moving_square, Drift_grid, Moving_dot
from predusion import im_processor as impor
from PIL import Image, ImageDraw
import hickle as hkl
import numpy as np
import os
import hickle as hkl
import matplotlib.pyplot as plt
from kitti_settings import *
from data_utils import SequenceGenerator
from scipy.misc import imresize

width = 200
step = 12
speed_list = range(1, 13) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image
init_pos = [30, width//2]
size_rect=20
nt = step

#out_data_head = 'moving_bar' + str(size_rect)

out_data_head = 'GratingStim'

train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')

## create raw video
#ms = Moving_square(width=width)
#
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)
#
train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)
X_train, label = train_generator.create_all(out_label=True)

###### check the video
one_video = X_train[0]
print(label)
im = one_video[0]

scale = 0.1

def rescale_im(im, scale=1):
    # create a black background
    im_bg = np.zeros(im.shape, dtype=np.uint8)
    scale_shape = (np.array(im.shape[:2] ) * scale).astype(np.uint8)
    # resize the image
    im_scale = imresize(im, size=tuple(scale_shape), interp='nearest')
    # add the image to a gray background
    center = np.array(im.shape[:2], dtype=np.uint8)//2
    center_coner = center - np.array(scale_shape, dtype=np.uint8) // 2
    im_bg[center_coner[0]: center_coner[0] + scale_shape[0], center_coner[1]: center_coner[1] + scale_shape[1], :] = im_scale
    return im_bg

for im in one_video:
    im_bg = rescale_im(im, scale=scale)
    print(np.max(im_bg), np.min(im_bg))
    plt.imshow(im_bg)
    plt.show()
