import os

## create dataset with out_data_head shown below
#os.system('python ./bin/gen_moving_bar_dataset.py')

#out_data_head_list = ['moving_bar20']
#out_data_head_list = ['moving_bar_obj_red']
#out_data_head_list = ['moving_bar_red']
#out_data_head_list = ['moving_arc']
#out_data_head_list = ['moving_text']
#out_data_head_list = ['moving_bar_y']
#out_data_head_list = ['moving_bar_wustl']

#nt = 11 # number of frames per video
#out_data_head_list = ['moving_bar_on_video'] # make sure nt equal to the number of frames

out_data_head_list = ['grating_stim', 'moving_bar_wustl', 'moving_rect2080'] # make sure nt equal to the number of frames
out_data_head_list = ['moving_bar20', 'moving_bar_red', 'moving_text'] # make sure nt equal to the number of frames

#out_data_head_list = ['dot_stim'] # make sure nt equal to the number of frames
#nt = 59
#out_data_head_list = ['mice_tf'] # make sure nt equal to the number of frames

#out_data_head_list = ['moving_rect2080'] # make sure nt equal to the number of frames

nt = 12

#weights_file = 'untrain_prednet_kitti_weights.hdf5'
#json_file = 'untrain_prednet_kitti_model.json'

weights_file = 'prednet_kitti_weights.hdf5'
json_file = 'prednet_kitti_model.json'

cut_time_geo = [3, nt] # analyze the geometric property only limit with this time interval. unit is frame
n_com_procrustes = 3

embed_method = 'pca'
cut_time_visual = [0, nt] # analyze the geometric property only limit with this time interval. unit is frame. used in visualize_neural_traj_speed

for out_data_head in out_data_head_list:
    os.system('python ./bin/neural_res_prednet.py --data_head ' + out_data_head + ' --nt ' + str(nt) + ' --weights_file ' + weights_file + ' --json_file ' + json_file)
    #os.system('python ./bin/moving_bar_geo.py --data_head ' + out_data_head + ' --nt ' + str(nt) + ' --cut_time ' + str(cut_time_geo[0]) + ' ' + str(cut_time_geo[1]) + ' --n_com_procrustes ' + str(n_com_procrustes))
    #os.system('python ./bin/visualize_neural_traj_speed.py --data_head ' + out_data_head + ' --nt ' + str(nt) + ' --cut_time ' + str(cut_time_visual[0]) + ' ' + str(cut_time_visual[1]) + ' --embed_method ' + embed_method)
