import os

nt = 12
# create dataset with out_data_head shown below
os.system('python ./bin/gen_moving_bar_dataset.py')


#out_data_head_list = ['moving_bar20']
#out_data_head_list = ['moving_bar_obj_red']
#out_data_head_list = ['moving_bar_red']
#out_data_head_list = ['moving_arc']
#out_data_head_list = ['moving_text']
#out_data_head_list = ['moving_bar_y']
#out_data_head_list = ['moving_bar_wustl']

#nt = 11 # number of frames per video
#out_data_head_list = ['moving_bar_on_video'] # make sure nt equal to the number of frames

#out_data_head_list = ['grating_stim'] # make sure nt equal to the number of frames
out_data_head_list = ['dot_stim'] # make sure nt equal to the number of frames
#weights_file = 'untrain_prednet_kitti_weights.hdf5'
#json_file = 'untrain_prednet_kitti_model.json'

#nt = 59
#out_data_head_list = ['mice_tf'] # make sure nt equal to the number of frames
weights_file = 'prednet_kitti_weights.hdf5'
json_file = 'prednet_kitti_model.json'

for out_data_head in out_data_head_list:
    os.system('python ./bin/gen_neural_res_moving_bar.py --data_head ' + out_data_head + ' --nt ' + str(nt) + ' --weights_file ' + weights_file + ' --json_file ' + json_file)
    #os.system('python ./bin/moving_bar_geo.py --data_head ' + out_data_head + ' --nt ' + str(nt))
    #os.system('python ./bin/visualize_neural_traj_speed.py --data_head ' + out_data_head + ' --nt ' + str(nt))
