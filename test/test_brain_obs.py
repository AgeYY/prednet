import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pprint
import copy
import hickle as hkl
from scipy import stats as st

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from kitti_settings import *
import predusion.allen_dataset as ad
from predusion.allen_data_generator import generate_allen_data

out_data_head = 'drifting_gratings' # currently we don't calculate the label yet

# load the BOC
drive_path = './data/allen-brain-observatory/visual-coding-2p'
manifest_file = os.path.join(drive_path, 'manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

# download one experiment
tuning_neuron_id = 3
exp_id = 511510699 # region VISp
exps = boc.get_ophys_experiments(experiment_container_ids=[exp_id], stimuli=['drifting_gratings'])
print("Experiments for experiment_container_id %d: %d\n" % (exp_id, len(exps)))
pprint.pprint(exps)

session_id = exps[0]['id'] # only one session has static_gratings, download it directly
data_set = boc.get_ophys_experiment_data(session_id)


ada = ad.Drifting_Gratings_Allen_Analyzer(data_set)
#
#dff, stim_epoch = ada.dff_trace()
#fig, ax = ad.plot_dff_trace(dff, stim_epoch)
#plt.show()
#
#for sid in range(3):
#    time_window, dff_stim = ada.single_stim(delta=30, stim_name ='drifting_gratings', stim_id=sid, neuron_id=3)
#    fig, ax = ad.plot_single_stim(time_window, dff_stim)
#    plt.show()
#
orivals, tfvals, tuning_array = ada.tuning(neuron_id=tuning_neuron_id)
for i in range(5):
    plt.plot(orivals, tuning_array[:,i], 'o-', label='{:.2f}'.format(tfvals[i]))
plt.legend()
plt.show()

feamap, label, label_name = generate_allen_data()

# calculate tuning curve
feamap = feamap['VISp']
feamap0 = feamap[:, tuning_neuron_id] # 0th neuron
cell_response = np.zeros( (label.shape[0], 3) )

for i in range(label.shape[0]):
    cell_response[i,0] = label[i, 0]
    cell_response[i,1] = label[i, 1]
    cell_response[i,2] = feamap0[i]

orivals = np.unique(label[:, 1])
orivals = orivals[np.isfinite(orivals)]

tfvals = np.unique(label[:, 0])
tfvals = tfvals[np.isfinite(tfvals)]

tuning_array = np.empty((8,5))
for i,ori in enumerate(orivals):
    for j,tf in enumerate(tfvals):
        trials = np.where(np.logical_and(cell_response[:,1]==ori, cell_response[:,0]==tf))[0]
        tuning_array[i,j] = cell_response[trials,2].mean()

for i in range(5):
    plt.plot(orivals, tuning_array[:,i], 'o-', label='{:.2f}'.format(tfvals[i]))
plt.legend()
plt.show()

#print(cell_response.head())
#
#unique_orientation = np.sort(cell_response['orientation'].dropna().unique())
#unique_temporal_frequency = np.sort(cell_response['temporal_frequency'].dropna().unique())
#
#def min_time_points(crs):
#    '''
#    crs (series, [n_trails]): each element has shape [n_cell, n_time], but the number of time points may not be the same. This function cut time points to make sure the number of time points in each trial are the same
#    '''
#    cell_response_same_tf = copy.deepcopy(crs)
#    min_n_time = 9999999
#    for cell_response_trial in cell_response_same_tf:
#        min_n_time = min(len(cell_response_trial[0]), min_n_time)
#
#    for i in range( len(cell_response_same_tf) ):
#        cell_response_same_tf.iloc[i] = np.array(cell_response_same_tf.iloc[i])[:, :min_n_time]
#
#    return cell_response_same_tf
#
#
## take average of orientation
#cell_response_tf = {tf: [] for tf in unique_temporal_frequency}
#for tf in cell_response_tf:
#    cell_response_same_tf = cell_response[ cell_response['temporal_frequency']==tf ]['response']
#    cell_response_same_tf = min_time_points(cell_response_same_tf)
#    cell_response_tf[tf] = cell_response_same_tf.mean()
#
##for cell in cell_response_tf[15.0]:
##    plt.plot(range(len(cell)), cell, color='grey')
##plt.show()
#
## the expected shape should be [n_trial, n_time, n_cells]
#n_time = cell_response_tf[15.0].shape[1]
#n_cell = cell_response_tf[15.0].shape[0]
#n_tf = len(cell_response_tf)
#
#response = np.empty((n_tf, n_time, n_cell))
#for i, tf in enumerate(cell_response_tf):
#    response[i] = np.nan_to_num(cell_response_tf[tf].T, nan=0)
#res_dic  = {'R0': response}
#
#hkl.dump(res_dic, os.path.join(DATA_DIR, output_name))
#
## in order to use the api of mvoing_bar_geo, we would like to create fake stimuli based on drifting grating
#
#from predusion.immaker import Moving_dot
#from predusion import im_processor as impor
#from PIL import Image, ImageDraw
#import hickle as hkl
#import numpy as np
#
#width = 200
#step = n_time
#speed_list = range(1, n_tf) # step = 12, speed_list_max = 12 make sure the squre doesn't fall out of the image
#
#out_data_head = 'mice_tf'
#category = out_data_head
#
#dg = Moving_dot()
#speed_list = np.linspace(0.02, 0.08, n_tf)
#sf = 0.02
#
#dg.set_stim_obj(obj_name='GratingStim', sf=sf)
#
#dg.create_video_batch(speed_list=speed_list, n_frame=step, category=out_data_head)
#categories = [out_data_head]
#impor.process_data(categories, out_data_head=out_data_head) #!!!!!!!!! WARNING the label may not be in the correct sequence as response
