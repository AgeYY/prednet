# generate moving bar images and dataset as moving_bar_train.hkl.

from predusion.immaker import Moving_square, Drift_grid, Moving_dot
from predusion import im_processor as impor
from PIL import Image, ImageDraw
import hickle as hkl
import numpy as np
from predusion.speed_data_generator import Dataset_generator

speed_list = range(1, 13) # step = 12, speed_list_max = 13 make sure the squre doesn't fall out of the image
size_rect=20
time_step = 8

dg = Dataset_generator()

#width, length = 20, 80
#out_data_head = 'moving_rect' + str(width) + str(length)
#dg.clear_image_folder(out_data_head)
#dg.moving_square(size_rect=(width, length), out_data_head=out_data_head, speed_list=speed_list, time_step=time_step)
#
#out_data_head = 'moving_bar' + str(size_rect)
#dg.clear_image_folder(out_data_head)
#dg.moving_square(size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list, time_step=time_step)
#
#out_data_head = 'moving_bar_red'
#dg.clear_image_folder(out_data_head)
#color_bag = (255, 0, 0)
#dg.moving_square(size_rect=size_rect, out_data_head=out_data_head, color_bag=color_bag, speed_list=speed_list, time_step=time_step)
#
#out_data_head = 'moving_bar_obj_red'
#dg.clear_image_folder(out_data_head)
#color_rect = (255, 0, 0)
#dg.moving_square(size_rect=size_rect, out_data_head=out_data_head, color_rect=color_rect, speed_list=speed_list, time_step=time_step)
#
#out_data_head = 'moving_arc'
#dg.clear_image_folder(out_data_head)
#dg.moving_arc(size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list, time_step=time_step)
#
#out_data_head = 'moving_text'
#dg.clear_image_folder(out_data_head)
#dg.moving_text(size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list, shape_para={'text': 'cool'}, time_step=time_step)
#
#out_data_head = 'moving_bar_wustl'
#dg.clear_image_folder(out_data_head)
#img_path = './data/wustl.jpg'
#dg.moving_square_image(img_path=img_path, size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list, time_step=time_step)
#
#out_data_head = 'moving_bar_on_video'
#dg.clear_image_folder(out_data_head)
#video_id = 6
#natural_video_path = './data/natural_video_' + str(video_id) + '.hkl'
#video = hkl.load(natural_video_path)
#dg.moving_square_on_video(video, size_rect=size_rect, time_step=time_step, out_data_head=out_data_head, speed_list=speed_list)

size=100
out_data_head = 'grating_stim'
dg.clear_image_folder(out_data_head)
speed_list = np.random.uniform(low=0.02, high=0.12, size=size)
ori_list = np.random.uniform(low=0, high=90, size=size)
scale=None
sf = 0.02
dg.grating_stim(sf=sf, time_step=time_step, out_data_head=out_data_head, speed_list=speed_list, ori_list=ori_list, scale=scale)
#
#scale=None
#out_data_head = 'dot_stim'
#dg.clear_image_folder(out_data_head)
#speed_list = np.linspace(1, 8, 12)
#coher = 0.7
#dg.dot_stim(coher=coher, time_step=time_step, out_data_head=out_data_head, speed_list=speed_list, scale=scale)
