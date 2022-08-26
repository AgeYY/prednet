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

out_data_head = 'drifting_grating' # currently we don't calculate the label yet

# load the BOC
drive_path = './data/allen-brain-observatory/visual-coding-2p'
manifest_file = os.path.join(drive_path, 'manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

# download one experiment
exp_id = 511510699 # region VISp
exps = boc.get_ophys_experiments(experiment_container_ids=[exp_id], stimuli=['drifting_gratings'])
print("Experiments for experiment_container_id %d: %d\n" % (exp_id, len(exps)))
pprint.pprint(exps)

session_id = exps[0]['id'] # only one session has static_gratings, download it directly
data_set = boc.get_ophys_experiment_data(session_id)


ada = ad.Drifting_Gratings_Allen_Analyzer(data_set)

dff, stim_epoch = ada.dff_trace()
fig, ax = ad.plot_dff_trace(dff, stim_epoch)
plt.show()

for sid in range(3):
    time_window, dff_stim = ada.single_stim(delta=30, stim_name ='drifting_gratings', stim_id=sid, neuron_id=3)
    fig, ax = ad.plot_single_stim(time_window, dff_stim)
    plt.show()

orivals, tfvals, tuning_array = ada.tuning()
for i in range(5):
    plt.plot(orivals, tuning_array[:,i], 'o-', label='{:.2f}'.format(tfvals[i]))
plt.legend()
plt.show()

## compare across trials with fixed condition ori, spatial, phase = (0, 0.02, 0)
#idx = (stim_table['orientation'] == 0.0) & (stim_table['spatial_frequency'] == 0.02) & (stim_table['phase'] == 0.00)
#t_delta = 5
#stim_table_subset = stim_table[idx].reset_index()
#
## get the temporal data
## record the number of time step in each trial
#trial = []
#time_len = []
#for i in range(len(stim_table_subset)):
#    trial.append(dff[:, stim_table_subset['start'][i] - t_delta: stim_table_subset['end'][i] + t_delta ] )
#    time_len.append(trial[-1].shape[1])
### find time_len of most trials, cut length longer than this, and remove length shorter than this. This works only when the time_len is highly cumulated at a single value, so other time lengths are just outlier. We recommand you to check distribution of time_len see whether the cut length is reasonable
### plot out the distribution of time_len
##plt.figure()
##plt.hist(time_len)
##plt.show()
#
#cut_len, _ = st.mode(time_len)
#print(cut_len[0])
#align_time_trial = []
#for tr in trial:
#    try:
#        align_time_trial.append(tr[:, :cut_len[0]])
#    except:
#        continue
#
#align_time_trial = np.array(align_time_trial) # (trial, neuron, time_len)
#
#for sn in range(30):
#    single_neuron = align_time_trial[:, sn, :]
#    plt.figure()
#    plt.imshow(single_neuron)
#    plt.show()

#fig = plt.figure(figsize=(10,8))
#for i in range(50): # show the first 50 DFF traces
#    plt.plot(dff[i]+(i*2), color='gray')
#plt.show()

#print(stim_table.head())
# get all dff for each stimulus

stim_name = 'drifting_gratings'
ts, dff = data_set.get_dff_traces()
stim_table = data_set.get_stimulus_table(stim_name)
stim_epoch = data_set.get_stimulus_epoch_table()

paraname = stim_table.keys().unique()
label_dic = {paraname[i]: [] for i in range(len(paraname))}
label_dic.pop('start', None)
label_dic.pop('end', None)
print(label_dic)

#label_dic = {'orientation': [], 'spatial_frequency': [], 'phase': []}
feamap = []
for i in range(len(stim_table)):
    for key in label_dic:
        label_dic[key].append(stim_table[key][i])
    res_cell = np.mean( dff[:, stim_table['start'][i]: stim_table['end'][i]], axis=1 ) # average response through the whole trial period
    feamap.append(res_cell)

label_name = list(label_dic.keys())
label = [label_dic[key] for key in label_dic]

label = np.array(label).T
feamap = {'VISp': np.array(feamap)}

neural_data_path = 'neural_' + out_data_head + '_allen_VISp' + '.hkl'
label_path = 'label_' + out_data_head + '_allen_VISp' + '.hkl'
label_name_path = 'label_name_' + out_data_head + '_allen_VISp' + '.hkl'

feamap_path = os.path.join(DATA_DIR, neural_data_path)
label_path = os.path.join(DATA_DIR, label_path)
label_name_path = os.path.join(DATA_DIR, label_name_path)

hkl.dump(feamap, feamap_path)
hkl.dump(label, label_path)
hkl.dump(label_name, label_name_path)

# check feamap and label

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
