# generate moving bar images and dataset as moving_bar_train.hkl.
import shutil

from predusion.immaker import Moving_square, Moving_dot
from predusion import im_processor as impor

from predusion.immaker import Moving_square, Drift_grid, Moving_dot
from predusion import im_processor as impor
from PIL import Image

class Dataset_generator():

    def clear_image_folder(self, out_data_head, save_dir_head='./kitti_data/raw/'):
        save_dir = save_dir_head + out_data_head + '/'
        try:
            shutil.rmtree(save_dir)
        except: pass

    def moving_square(self, size_rect=20, width=200, time_step=12, init_pos=[30, 100], scale=None, out_data_head = 'moving_bar', color_bag=(0, 0, 0), color_rect=(255, 255, 255), speed_list=[0]):
        '''
        input:
          width (int): the shape of each frame is width * width. The width should be larger than the input of the predent
          size_rect (int, or (width, length)): size of the square
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

    def moving_square_on_video(self, video, size_rect=30, width=200, time_step=12, init_pos=[30, 100], scale=None, out_data_head = 'moving_bar_video', speed_list=[0]):
        '''
        time_step (int): the resulting videos would only have time step = min(time_step, number of frames in the input video)
        '''
        ms = Moving_square(width=width)

        ms.create_video_batch_on_video(video, init_pos=init_pos, speed=speed_list, step=time_step, size_rect=size_rect, category=out_data_head)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head)

    def grating_stim(self, sf=0.02, width=200, time_step=12, scale=None, out_data_head = 'grating_stim', speed_list=[0], ori_list=[0]):
        if scale is None:
            size_frame = width
        else:
            size_frame = int(scale * width)

        dg = Moving_dot(imshape=(width, width))
        dg.set_stim_obj(obj_name='GratingStim', sf=sf, size=size_frame)

        dg.create_video_batch_grating_stim(speed_list=speed_list, ori_list=ori_list, n_frame=time_step, category=out_data_head)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head)

    def dot_stim(self, coher=0.7, width=200, time_step=12, scale=None, out_data_head = 'dot_stim', speed_list=[0]):
        #if scale is None:
        #    size_frame = width
        #else:
        #    size_frame = int(scale * width)

        dg = Moving_dot(imshape=(width, width))
        dg.set_stim_obj(obj_name='DotStim', coherence=coher)
        
        dg.create_video_batch(speed_list=speed_list, n_frame=time_step, category=out_data_head)
        categories = [out_data_head]
        impor.process_data(categories, out_data_head=out_data_head)
