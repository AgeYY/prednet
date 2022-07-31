# only test the grating_stim
from predusion.immaker import Moving_square, Drift_grid, Moving_dot
from predusion import im_processor as impor
from PIL import Image, ImageDraw
import hickle as hkl
import numpy as np
from predusion.speed_data_generator import Dataset_generator

out_data_head = 'grating_stim'
dg = Dataset_generator()

time_step = 8
size = 100
#speed_list = np.linspace(0.02, 0.12, 12)
#ori_list = np.linspace(0, 360, 12, endpoint=False)
speed_list = np.random.uniform(low=0.02, high=0.12, size=size)
ori_list = np.random.uniform(low=0, high=90, size=size)
#ori_list = []
scale=None
sf = 0.02
dg.clear_image_folder(out_data_head)
dg.grating_stim(sf=sf, time_step=time_step, out_data_head=out_data_head, speed_list=speed_list, ori_list=ori_list, scale=scale)

x_train = hkl.load('kitti_data/' + out_data_head + '_X_train.hkl')
print(x_train.shape)

source_train = hkl.load('kitti_data/' + out_data_head + '_sources_train.hkl')
print(len(source_train))


label = hkl.load('kitti_data/' + out_data_head + '_label.hkl')
print(label)

label_name = hkl.load('kitti_data/' + out_data_head + '_label_name.hkl')
print(label_name)

##### output neural response
## generate prednet neural response to moving_bar_train.hkl which is generated from gen_moving_bar_dataset.py
#import os
#import hickle as hkl
#import matplotlib.pyplot as plt
#import numpy as np
#
#from predusion.agent import Agent
#from data_utils import SequenceGenerator, convert_prednet_output
#from kitti_settings import *
#
#nt = 12
#weights_file = 'prednet_kitti_weights.hdf5'
#json_file = 'prednet_kitti_model.json'
#
#batch_size = None # number of predicting videos in each batch. Doesn't matter
#
#weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)
#json_file = os.path.join(WEIGHTS_DIR, json_file)
#train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
#train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
#label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
#label_name_file = os.path.join(DATA_DIR, out_data_head + '_label_name.hkl')
##output_mode = ['E0', 'E1', 'E2', 'E3']
#output_mode = ['R0']
#output_neural = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
#output_label = 'label_' + out_data_head + '_R_prednet' + '.hkl'
#output_label_name = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'
#
###
#train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, label_name_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)
#
#X_train, label, label_name = train_generator.create_all(out_label=True)
#
#sub = Agent()
#sub.read_from_json(json_file, weights_file)
#
#output = sub.output_multiple(X_train, output_mode=output_mode, batch_size=batch_size, is_upscaled=False)
#output['X'] = X_train
#
## flatten features, and add time label to each video frame
#output, label, label_name = convert_prednet_output(output, np.array(label), label_name)
#print(label_name)
