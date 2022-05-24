# generate moving bar images and dataset as moving_bar_train.hkl.

from predusion.immaker import Moving_square
from predusion import im_processor as impor
from PIL import Image, ImageDraw
import hickle as hkl

width = 200
step = 12
speed_list = range(1, 13) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image
init_pos = [30, width//2]

# create raw video
ms = Moving_square(width=width)

#size_rect=10
#out_data_head = 'moving_bar' + str(size_rect)
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#size_rect=20
#out_data_head = 'moving_bar_bg_color'
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head, color_bag=(255, 0, 0))
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)
#
#size_rect=20
#out_data_head = 'moving_bar_obj_color'
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head, color_rect=(255, 0, 0))
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#size_rect=20
#out_data_head = 'moving_arc'
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head, shape='arc', shape_para={'start': 0, 'end': 120})
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#size_rect=30
#out_data_head = 'moving_text'
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head, shape='text', shape_para={'text': 'hello'})
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#size_rect=30
#out_data_head = 'moving_bar_y'
#ms.create_video_batch(init_pos=[30, width * 3 //4], speed=speed_list, step=step, size_rect=size_rect, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

#size_rect=30
#out_data_head = 'moving_bar_wustl'
#img_bag = Image.open('./data/wustl.jpg').convert('RGB').resize((width, width))
#
#ms.init_im(img_bag)
#ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head)

size_rect=30
out_data_head = 'moving_bar_on_video'
video_id = 6
natural_video_path = './data/natural_video_' + str(video_id) + '.hkl'

video = hkl.load(natural_video_path)
ms.create_video_batch_on_video(video, init_pos=init_pos, speed=speed_list, step=step, size_rect=size_rect, category=out_data_head)
categories = [out_data_head]
impor.process_data(categories, out_data_head=out_data_head)
