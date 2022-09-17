import os
import numpy as np
import hickle as hkl
import pprint
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import predusion.allen_dataset as ad

from kitti_settings import *

class Allen_Generator():
    def __init__(self):
        ''' you can use boc to inqury like exp id or something'''
        manifest_file = os.path.join(ALLEN_DRIVE_PATH, 'manifest.json')
        self.boc = BrainObservatoryCache(manifest_file=manifest_file)

    def generate(self, exp_id=511510699, stimuli='drifting_gratings', out_data_head='drifting_gratings', verbose=True):
        '''
        Load Allen data and convert to feamap format. Current version works on drifting_gratings only
        Always select the first session contains target simuli
        '''
        self.exp_id = exp_id
        self.stimuli = stimuli

        self.exps = self.boc.get_ophys_experiments(experiment_container_ids=[self.exp_id], stimuli=[self.stimuli])

        if verbose:
            pprint.pprint(self.exps)

        session_id = self.exps[0]['id'] # only one session has static_gratings, download it directly
        self.data_set = self.boc.get_ophys_experiment_data(session_id)

        targeted_structure = self.exps[0]['targeted_structure']
        ts, dff = self.data_set.get_dff_traces()
        stim_table = self.data_set.get_stimulus_table(out_data_head)

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

        self.label_name = list(label_dic.keys())
        self.label = np.array([label_dic[key] for key in label_dic])

        self.label = np.array(self.label).T
        self.feamap = {targeted_structure: np.array(feamap)}

        neural_data_path = 'neural_' + out_data_head + '_allen_VISp' + '.hkl'
        label_path = 'label_' + out_data_head + '_allen_VISp' + '.hkl'
        label_name_path = 'label_name_' + out_data_head + '_allen_VISp' + '.hkl'

        feamap_path = os.path.join(DATA_DIR, neural_data_path)
        label_path = os.path.join(DATA_DIR, label_path)
        label_name_path = os.path.join(DATA_DIR, label_name_path)

        hkl.dump(self.feamap, feamap_path)
        hkl.dump(self.label, label_path)
        hkl.dump(self.label_name, label_name_path)

        return self.feamap.copy(), self.label.copy(), self.label_name.copy()

    def tuning(self, neuron_index=None, cell_id=None, verbose=False):
        '''
        if both None --> error
        one is None --> use the another one
        both not None --> use neuron_index
        '''

        if neuron_index is None:
            neuron_index = self.cell_id_2_index(cell_id)[0]
        else:
            cell_id = self.cell_index_2_id(neuron_index)[0]

        # calculate tuning curve
        feamap = self.feamap['VISp']
        feamap0 = feamap[:, neuron_index] # 0th neuron
        cell_response = np.zeros( (self.label.shape[0], 3) )

        for i in range(self.label.shape[0]):
            cell_response[i,0] = self.label[i, 0]
            cell_response[i,1] = self.label[i, 1]
            cell_response[i,2] = feamap0[i]

        orivals = np.unique(self.label[:, 1])
        orivals = orivals[np.isfinite(orivals)]

        tfvals = np.unique(self.label[:, 0])
        tfvals = tfvals[np.isfinite(tfvals)]

        tuning_array = np.empty((8,5))
        for i,ori in enumerate(orivals):
            for j,tf in enumerate(tfvals):
                trials = np.where(np.logical_and(cell_response[:,1]==ori, cell_response[:,0]==tf))[0]
                tuning_array[i,j] = cell_response[trials,2].mean()
        if verbose:
            return orivals, tfvals, tuning_array, neuron_index, cell_id
        else:
            return orivals, tfvals, tuning_array

    def cell_index_2_id(self, cell_index):
        cell_id = self.data_set.get_cell_specimen_ids()[cell_index]

        if isinstance(cell_id, list):
            cell_id_list = cell_id
        else:
            cell_id_list = [cell_id]

        return cell_id_list

    def cell_id_2_index(self, cell_id):
        '''
        cell_id: int or list
        '''
        if isinstance(cell_id, list):
            cell_id_list = cell_id
        else:
            cell_id_list = [cell_id]

        cell_index = self.data_set.get_cell_specimen_indices(cell_specimen_ids=cell_id_list)
        return cell_index

