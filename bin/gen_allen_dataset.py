# generate neural data from allen, and do some simple analysis for fast checking. test/test_brain_obs.py has demo of using predusion.allen_dataset

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
from predusion.allen_data_generator import Allen_Generator

out_data_head = 'drifting_gratings' # currently we don't calculate the label yet
stimuli = 'drifting gratings'
exp_id = 511510699 # region VISp
neuron_index = 0
cell_id =  517517168

allen = Allen_Generator()
feamap, label, label_name = allen.generate(exp_id)

orivals, tfvals, tuning_array, neuron_index, cell_id = allen.tuning(cell_id=cell_id, verbose=True)
print(neuron_index, cell_id)

for i in range(5):
    plt.plot(orivals, tuning_array[:,i], 'o-', label='{:.2f}'.format(tfvals[i]))
plt.legend()
plt.show()
