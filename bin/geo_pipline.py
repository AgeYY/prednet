import os

nt = 11
os.system('python ./bin/gen_moving_bar_dataset.py')

#out_data_head_list = ['moving_bar_bg_color', 'moving_bar_obj_color']
#out_data_head_list = ['moving_arc']
#out_data_head_list = ['moving_text']
#out_data_head_list = ['moving_bar_y']
#out_data_head_list = ['moving_bar_wustl']
out_data_head_list = ['moving_bar_on_video'] # make sure nt equal to the number of frames
#out_data_head_list = ['GratingStim'] # make sure nt equal to the number of frames
#out_data_head_list = ['DotStim1.0', 'DotStim0.8', 'DotStim0.5', 'DotStim0.3'] # make sure nt equal to the number of frames
for out_data_head in out_data_head_list:
    os.system('python ./bin/gen_neural_res_moving_bar.py --data_head ' + out_data_head + ' --nt ' + str(nt))
    os.system('python ./bin/moving_bar_geo.py --data_head ' + out_data_head + ' --nt ' + str(nt))
    os.system('python ./bin/visualize_neural_traj_speed.py --data_head ' + out_data_head + ' --nt ' + str(nt))
