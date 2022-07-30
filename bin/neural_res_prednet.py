# generate prednet neural response to moving_bar_train.hkl which is generated from gen_moving_bar_dataset.py
import os
import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np

from predusion.agent import Agent
from data_utils import SequenceGenerator, convert_prednet_output
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

batch_size = 10 # number of predicting videos in each batch. Doesn't matter

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)
json_file = os.path.join(WEIGHTS_DIR, json_file)
train_file = os.path.join(DATA_DIR, out_data_head + '_X_train.hkl')
train_sources = os.path.join(DATA_DIR, out_data_head + '_sources_train.hkl')
label_file = os.path.join(DATA_DIR, out_data_head + '_label.hkl')
label_name_file = os.path.join(DATA_DIR, out_data_head + '_label_name.hkl')
#output_mode = ['E0', 'E1', 'E2', 'E3']
output_mode = ['R0', 'R1', 'R2', 'R3']
output_neural = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
output_label = 'label_' + out_data_head + '_R_prednet' + '.hkl'
output_label_name = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

##
train_generator = SequenceGenerator(train_file, train_sources, nt, label_file, label_name_file, sequence_start_mode='unique', output_mode='prediction', shuffle=False)

X_train, label, label_name = train_generator.create_all(out_label=True)

sub = Agent()
sub.read_from_json(json_file, weights_file)

output = sub.output_multiple(X_train, output_mode=output_mode, batch_size=batch_size, is_upscaled=False)
output['X'] = X_train

# flatten features, and add time label to each video frame
output, label, label_name = convert_prednet_output(output, label, label_name)

hkl.dump(output, os.path.join(DATA_DIR, output_neural))
hkl.dump(label, os.path.join(DATA_DIR, output_label))
hkl.dump(label_name, os.path.join(DATA_DIR, output_label_name))
print(label_name)

#Check the prediction
import matplotlib.pyplot as plt
import numpy as np
from predusion.ploter import Ploter

output = sub.output(X_train, output_mode='prediction', batch_size=batch_size, is_upscaled=False)
#idx = (np.array(label, dtype=int) == 12).nonzero()[0][0] # show square moving with speed = 12. This could be used to check the label
idx = -1
plter = Ploter()
fig, gs = plter.plot_seq_prediction(X_train[idx], output[idx])
plt.show()

