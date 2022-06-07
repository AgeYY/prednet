import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pprint
import copy
import hickle as hkl

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from kitti_settings import *

output_name = 'neural_moving_bar_R' + '.hkl'

drive_path = '../data/allen-brain-observatory/visual-coding-2p'
# drive_path = ''
manifest_file = os.path.join(drive_path,'manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

# download one experiment
exp_id = 511510699
exps = boc.get_ophys_experiments(experiment_container_ids=[exp_id], stimuli=['drifting_gratings'])
print("Experiments for experiment_container_id %d: %d\n" % (exp_id, len(exps)))
pprint.pprint(exps)

session_id = exps[0]['id']
data_set = boc.get_ophys_experiment_data(session_id)

ts, dff = data_set.get_dff_traces()
#print(dff.shape)

fig = plt.figure(figsize=(10,8))
#for i in range(50):
#    plt.plot(dff[i]+(i*2), color='gray')
#plt.show()

stim_table = data_set.get_stimulus_table('drifting_gratings')

#print(stim_table.head())

cell_response = {'orientation': [], 'temporal_frequency': [], 'response': []}
for i in range(len(stim_table)):
    cell_response['orientation'].append(stim_table['orientation'][i])
    cell_response['temporal_frequency'].append(stim_table['temporal_frequency'][i])
    res_cell = dff[:, stim_table['start'][i]: stim_table['end'][i]]
    cell_response['response'].append(res_cell)
cell_response = pd.DataFrame(cell_response)
#print(cell_response.head())

unique_orientation = np.sort(cell_response['orientation'].dropna().unique())
unique_temporal_frequency = np.sort(cell_response['temporal_frequency'].dropna().unique())

def min_time_points(crs):
    '''
    crs (series, [n_trails]): each element has shape [n_cell, n_time], but the number of time points may not be the same. This function cut time points to make sure the number of time points in each trial are the same
    '''
    cell_response_same_tf = copy.deepcopy(crs)
    min_n_time = 9999999
    for cell_response_trial in cell_response_same_tf:
        min_n_time = min(len(cell_response_trial[0]), min_n_time)

    for i in range( len(cell_response_same_tf) ):
        cell_response_same_tf.iloc[i] = np.array(cell_response_same_tf.iloc[i])[:, :min_n_time]

    return cell_response_same_tf


# take average of orientation
cell_response_tf = {tf: [] for tf in unique_temporal_frequency}
for tf in cell_response_tf:
    cell_response_same_tf = cell_response[ cell_response['temporal_frequency']==tf ]['response']
    cell_response_same_tf = min_time_points(cell_response_same_tf)
    cell_response_tf[tf] = cell_response_same_tf.mean()

#for cell in cell_response_tf[15.0]:
#    plt.plot(range(len(cell)), cell, color='grey')
#plt.show()

# the expected shape should be [n_trial, n_time, n_cells]
n_time = cell_response_tf[15.0].shape[1]
n_cell = cell_response_tf[15.0].shape[0]
n_tf = len(cell_response_tf)

response = np.empty((n_tf, n_time, n_cell))
for i, tf in enumerate(cell_response_tf):
    response[i] = np.nan_to_num(cell_response_tf[tf].T, nan=0)
res_dic  = {'R0': response}

hkl.dump(res_dic, os.path.join(DATA_DIR, output_name))

# in order to use the api of mvoing_bar_geo, we would like to create fake stimuli based on drifting grating

from predusion.immaker import Moving_dot
from predusion import im_processor as impor
from PIL import Image, ImageDraw
import hickle as hkl
import numpy as np

width = 200
step = n_time
speed_list = range(1, n_tf) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image

out_data_head = 'mice_tf'
category = out_data_head

dg = Moving_dot()
speed_list = np.linspace(0.02, 0.08, n_tf)
sf = 0.02

dg.set_stim_obj(obj_name='GratingStim', sf=sf)

dg.create_video_batch(speed_list=speed_list, n_frame=step, category=out_data_head)
categories = [out_data_head]
impor.process_data(categories, out_data_head=out_data_head) #!!!!!!!!! WARNING the label may not be in the correct sequence as response
