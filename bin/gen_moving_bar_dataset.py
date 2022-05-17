# generate moving bar images and dataset as moving_bar_train.hkl.

from predusion.immaker import Moving_square

width = 200
step = 12
speed_list = range(1, 13) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image
init_pos = [30, width//2]

# create raw video
ms = Moving_square(width=width)

ms.create_video_batch(init_pos=init_pos, speed=speed_list, step=step)

from predusion import im_processor as impor
categories = ['moving_bar']
out_data_head = 'moving_bar'
impor.process_data(categories, out_data_head=out_data_head)
