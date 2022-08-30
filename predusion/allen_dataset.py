import matplotlib.pyplot as plt
import numpy as np

class Drifting_Gratings_Allen_Analyzer():
    ''' This class works on boc.get_ophys_experiment_data'''
    def __init__(self, data_set):
        '''
        get ts and dff by ts, dff = data_set.get_dff_traces()
        get stim_table by stim_table = data_set.get_stimulus_table('static_gratings'). 'static_gratings' can be replaced to others
        '''
        self.data_set = data_set

        #self.ts = ts
        #self.dff = dff # n_neuron, t_step
        #self.stim_table = stim_table

    def dff_trace(self):
        ts, dff = self.data_set.get_dff_traces()
        stim_epoch = self.data_set.get_stimulus_epoch_table()
        return dff, stim_epoch

    def single_stim(self, delta=30, stim_name ='drifting_gratings', stim_id=0, neuron_id=0):
        stim_table = self.data_set.get_stimulus_table(stim_name)
        ts, dff = self.data_set.get_dff_traces()
        dff_stim = dff[neuron_id, stim_table.start[stim_id]-delta:stim_table.end[stim_id]+delta]
        time_window = [delta, dff_stim.shape[0] - delta]

        return time_window, dff_stim

    def tuning(self, neuron_id=0):
        ts, dff = self.data_set.get_dff_traces()
        dff_trace = dff[neuron_id]

        stim_table = self.data_set.get_stimulus_table('drifting_gratings')
        cell_response= np.zeros((len(stim_table),3))
        for i in range(len(stim_table)):
            cell_response[i,0] = stim_table.orientation[i]
            cell_response[i,1] = stim_table.temporal_frequency[i]
            cell_response[i,2] = dff_trace[stim_table.start[i]:stim_table.end[i]].mean()

        all_ori = np.unique(cell_response[:,0])
        orivals = all_ori[np.isfinite(all_ori)]

        tfvals = np.unique(cell_response[:,1])
        tfvals = tfvals[np.isfinite(tfvals)]

        tuning_array = np.empty((8,5))
        for i,ori in enumerate(orivals):
            for j,tf in enumerate(tfvals):
                trials = np.where(np.logical_and(cell_response[:,0]==ori, cell_response[:,1]==tf))[0]
                tuning_array[i,j] = cell_response[trials,2].mean()
        return orivals, tfvals, tuning_array

def plot_dff_trace(dff, stim_epoch, n_neuron=50):
    ### plot dff trace, maximum 5 stim_epoch
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(n_neuron): # show the first 50 DFF traces
        ax.plot(dff[i]+(i*2), color='gray')

    paraname = stim_epoch.stimulus.unique()
    #for each stimulus, shade the plot when the stimulus is presented
    colors = ['blue', 'orange', 'green', 'red', 'yellow']
    color_dic = {paraname[i]: colors[i] for i in range(len(paraname))}

    for c,stim_name in enumerate(paraname):
        stim = stim_epoch[stim_epoch.stimulus==stim_name]
        for j in range(len(stim)):
            ax.axvspan(xmin=stim.start.iloc[j], xmax=stim.end.iloc[j], color=color_dic[stim_name], alpha=0.1)

    return fig, ax

def plot_single_stim(time_window, dff_stim):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.plot(dff_stim)
    ax.axvspan(*time_window, color='gray', alpha=0.3) #this shades the period when the stimulus is being presented
    ax.set_ylabel("DF/F")
    ax.set_xlabel("Frames")
    return fig, ax
