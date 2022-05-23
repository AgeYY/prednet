# generate prednet neural response to moving_bar_train.hkl which is generated from gen_moving_bar_dataset.py
import os
import hickle as hkl
import matplotlib.pyplot as plt

from predusion.agent import Agent
from data_utils import SequenceGenerator
from kitti_settings import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_head', default='moving_bar', type=str,
                    help='head of the dataset')
arg = parser.parse_args()

out_data_head = arg.data_head

nt = 12 # each prediction group contains nt images
batch_size = 10

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
output_mode = ['E0', 'E1', 'E2', 'E3']
output_name = 'neural_' + out_data_head + '_E' + '.hkl'
#output_mode = ['R0', 'R1', 'R2', 'R3']
#output_name = 'neural_moving_bar_R' + '.hkl'

##
train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)

X_train, label = train_generator.create_all(out_label=True)

####### check the video
#one_video = X_train[-1]
#for im in one_video:
#    plt.imshow(im)
#    plt.show()

sub = Agent()
sub.read_from_json(json_file, weights_file)

output = sub.output_multiple(X_train, output_mode=output_mode, batch_size=batch_size)

hkl.dump(output, os.path.join(DATA_DIR, output_name))
