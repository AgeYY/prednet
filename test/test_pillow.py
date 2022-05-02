from predusion.immaker import Moving_square

width = 200
step = 12
speed_list = range(0, 12) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image

ms = Moving_square(width=width)
for sp in speed_list:
    ms.create_video(init_pos = [30, width//2], speed=sp, step=step)
    ms.save_image(save_dir_label='moving_bar/sp_' + str(sp) + '/')
    ms.clear_image()
