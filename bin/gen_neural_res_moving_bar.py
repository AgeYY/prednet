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
parser.add_argument('--nt', default=12, type=int,
                    help='number of frames per video')
parser.add_argument('--weights_file', default='prednet_kitti_weights.hdf5', type=str,
                    help='weights for prednet')
parser.add_argument('--json_file', default='prednet_kitti_model.json', type=str,
                    help='json for the prednet')

arg = parser.parse_args()

out_data_head = arg.data_head
nt = arg.nt # number of time points
weights_file = arg.weights_file
json_file = arg.json_file

batch_size = nt # what is batch size?

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)
json_file = os.path.join(WEIGHTS_DIR, json_file)
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
#output_mode = ['E0', 'E1', 'E2', 'E3']
#output_name = 'neural_' + out_data_head + '_E' + '.hkl'
output_mode = ['R0', 'R1', 'R2', 'R3']
output_name = 'neural_moving_bar_R' + '.hkl'

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
#
#output = sub.output_multiple(X_train, output_mode=output_mode, batch_size=batch_size, is_upscaled=False)

#Check the prediction
import matplotlib.pyplot as plt
from predusion.ploter import Ploter
output = sub.output(X_train, output_mode='prediction', batch_size=batch_size, is_upscaled=False)
plter = Ploter()
fig, gs = plter.plot_seq_prediction(X_train[-1], output[-1])
plt.show()

hkl.dump(output, os.path.join(DATA_DIR, output_name))
