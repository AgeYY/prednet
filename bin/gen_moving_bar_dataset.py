# generate moving bar images and dataset as moving_bar_train.hkl.

from predusion.immaker import Moving_square, Drift_grid, Moving_dot
from predusion import im_processor as impor
from PIL import Image, ImageDraw
import hickle as hkl
import numpy as np

class Dataset_generator():
    def moving_square(self, size_rect=20, width=200, time_step=12, init_pos=[30, 100], scale=None, out_data_head = 'moving_bar', color_bag=(0, 0, 0), color_rect=(255, 255, 255), speed_list=[0]):
        '''
        input:
          width (int): the shape of each frame is width * width. The width should be larger than the input of the predent
          size_rect (int): size of the square
          time_step (int): the number of frames contained in one movie
          init_pos (list of int): initial position of the square
          scale (float): move the video to the center of each frame with width equal to int(width * scale)
          out_data_head (str): the name head of the data
          color_bag (tuple [3_channels]): backgroud color
          color_rect (tuple [3_channels]): object color
          speed_list (array [n_speeds]): speeds for the video batch
        output:
          videos are stored in ./kitti_data/raw/$out_data_head$/sp_i/
          labels for each video is stored in ./kitti_data/raw/$out_data_head$/label.json. It is a dictionary {$out_data_head$-sp_i: speed value}
          processed video file: The processing procedure including reshape, scale (if scale is not None), and make the data readable by SequenceGenerator. Please check process_data function for more detail.
        '''
        ms = Moving_square(width=width)

        ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=time_step, size_rect=size_rect, category=out_data_head, color_bag=color_bag, color_rect=color_rect)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head, scale=scale)

    def moving_arc(self, size_rect=20, width=200, time_step=12, init_pos=[30, 100], scale=None, out_data_head = 'moving_arc', speed_list=[0], shape_para={'start': 0, 'end': 120}):
        '''
        input:
          shape_para (dict): start and end angle of the arc
          size_rect (int): the box contains the arc (or the size of the arc)
        '''

        ms = Moving_square(width=width)

        ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=time_step, size_rect=size_rect, category=out_data_head, shape='arc', shape_para=shape_para)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head, scale=scale)

    def moving_text(self, size_rect=20, width=200, time_step=12, init_pos=[30, 100], scale=None, out_data_head = 'moving_text', speed_list=[0], shape_para={'text': 'hello'}):
        '''
        input:
          shape_para (dict): start and end angle of the arc
          size_rect (int): the box contains the arc (or the size of the arc)
        '''

        ms = Moving_square(width=width)

        ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=time_step, size_rect=size_rect, category=out_data_head, shape='text', shape_para=shape_para)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head, scale=scale)

    def moving_square_image(self, img_path, size_rect=20, width=200, time_step=12, init_pos=[30, 100], scale=None, out_data_head = 'moving_bar_image', speed_list=[0]):
        '''
        a square moving from left to right on a static background
        '''
        ms = Moving_square(width=width)
        img_bag = Image.open(img_path).convert('RGB').resize((width, width))

        ms.init_im(img_bag)
        ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=time_step, size_rect=size_rect, category=out_data_head)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head, scale=scale)


speed_list = range(1, 13) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image
size_rect=20

dg = Dataset_generator()

out_data_head = 'moving_bar' + str(size_rect)
dg.moving_square(size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list)

out_data_head = 'moving_bar_red'
color_bag = (255, 0, 0)
dg.moving_square(size_rect=size_rect, out_data_head=out_data_head, color_bag=color_bag, speed_list=speed_list)

out_data_head = 'moving_bar_obj_red'
color_rect = (255, 0, 0)
dg.moving_square(size_rect=size_rect, out_data_head=out_data_head, color_rect=color_rect, speed_list=speed_list)

out_data_head = 'moving_arc'
dg.moving_arc(size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list)

out_data_head = 'moving_text'
dg.moving_text(size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list, shape_para={'text': 'cool'})

out_data_head = 'moving_bar_wustl'
img_path = './data/wustl.jpg'
dg.moving_square_image(img_path=img_path, size_rect=size_rect, out_data_head=out_data_head, speed_list=speed_list)

#size_rect=30
#out_data_head = 'moving_bar_on_video'
#video_id = 6
#natural_video_path = './data/natural_video_' + str(video_id) + '.hkl'
#
#video = hkl.load(natural_video_path)
#ms.create_video_batch_on_video(video, init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#out_data_head = 'GratingStim'
#category = out_data_head
#
#dg = Moving_dot()
#speed_list = np.linspace(0.02, 0.08, step)
#sf = 0.08
#size = 15
#
#dg.set_stim_obj(obj_name=out_data_head, sf=sf, size=size)
#
#dg.create_video_batch(speed_list=speed_list, n_frame=step, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#dg = Moving_dot()
#speed_list = np.linspace(1, 8, step)
#
#for coher in [1.0, 0.8, 0.5, 0.3]:
#    out_data_head = 'DotStim' + str(coher)
#    dg.set_stim_obj(obj_name='DotStim', coherence=coher)
#    dg.create_video_batch(speed_list=speed_list, n_frame=step, category=out_data_head)
#    categories = [out_data_head]
#    impor.process_data(categories, out_data_head=out_data_head)
