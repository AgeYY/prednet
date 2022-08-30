import os
import numpy as np
import hickle as hkl
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import predusion.allen_dataset as ad

from kitti_settings import *

def generate_allen_data(exp_id=511510699, stimuli='drifting_gratings', out_data_head='drifting_gratings'):
    '''
    Load Allen data and convert to feamap format. Current version works on drifting_gratings only
    Always select the first session contains target simuli
    '''
    manifest_file = os.path.join(ALLEN_DRIVE_PATH, 'manifest.json')
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    exps = boc.get_ophys_experiments(experiment_container_ids=[exp_id], stimuli=[stimuli])
    session_id = exps[0]['id'] # only one session has static_gratings, download it directly
    data_set = boc.get_ophys_experiment_data(session_id)

    targeted_structure = exps[0]['targeted_structure']
    ts, dff = data_set.get_dff_traces()
    stim_table = data_set.get_stimulus_table(out_data_head)

    paraname = stim_table.keys().unique() # parameters in stimuli
    label_dic = {paraname[i]: [] for i in range(len(paraname))}
    label_dic.pop('start', None)
    label_dic.pop('end', None)
    label_dic.pop('blank_sweep', None)

    stim_table = stim_table[stim_table.blank_sweep==0.0].reset_index().drop(columns=['index', 'blank_sweep']) # only pick non_blank

    feamap = []
    for i in range(len(stim_table)):
        for key in label_dic:
            label_dic[key].append(stim_table[key][i])
        res_cell = np.mean( dff[:, stim_table['start'][i]: stim_table['end'][i]], axis=1 ) # average response through the whole trial period
        feamap.append(res_cell)

    label_name = list(label_dic.keys())
    label = [label_dic[key] for key in label_dic]

    label = np.array(label).T
    feamap = {targeted_structure: np.array(feamap)}

    neural_data_path = 'neural_' + out_data_head + '_allen_VISp' + '.hkl'
    label_path = 'label_' + out_data_head + '_allen_VISp' + '.hkl'
    label_name_path = 'label_name_' + out_data_head + '_allen_VISp' + '.hkl'

    feamap_path = os.path.join(DATA_DIR, neural_data_path)
    label_path = os.path.join(DATA_DIR, label_path)
    label_name_path = os.path.join(DATA_DIR, label_name_path)

    hkl.dump(feamap, feamap_path)
    hkl.dump(label, label_path)
    hkl.dump(label_name, label_name_path)

    return feamap, label, label_name
