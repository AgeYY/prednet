import numpy as np

def tensor_sta(fir_rate, stimuli, n_tao):
    '''
    spikes triggered average.
    input:
      fir_rate (array like, float, [n_time_step, ...])
      stimuli (array like, float, [n_time_step, ...]): should have the same n_time_step as fir_rate
      n_tao (int): how long the sta you want
    output:
      qij (numpy array, float, [n_tao, *fir_rate.shape[1:], *stimuli.shape[1:]]): the tensor dot of the fir_rate and stimuli
    '''
    fir_rate_proc = fir_rate[n_tao:] # avoid end effect
    norm = np.sum(fir_rate_proc, axis=0)
    try:
        norm[np.abs(norm) < 1e-10] = 1 # we don't care the norm is no firing rate, the firing rate are all 0s.
    except:
        pass
    fir_rate_proc = fir_rate_proc / norm
    fir_rate_n_time = fir_rate_proc.shape[0]
    qij = [] # the correlation function
    for tao in range(n_tao):
        rf_tao = np.tensordot(fir_rate_proc, stimuli[n_tao - tao:n_tao - tao + fir_rate_n_time], axes=([0], [0]))
        qij.append(rf_tao.copy())
    qij = np.array(qij) # [number of tao time point, number of neurons, *image_shape, 3]
    return qij

def confidence_interval(data, measure=np.mean, n_resamples=None, size=None, ci_bound=[2.5, 97.5]):
    '''
    find the confidence interval of measure(data)
    data (np.array, [n_data_points])
    meansure: a function whose input is an array output is a number. measure(data)
    '''
    if n_resamples is None:
        n_resamples = data.shape[0]
    if size is None:
        size = data.shape[0]

    measure_arr = np.zeros(n_resamples)

    idx_data = np.arange(data.shape[0])
    for i in range(n_resamples):
        idx_resample = np.random.choice(idx_data, replace=True, size=size) # the batch size is according to https://stats.stackexchange.com/questions/246726/size-of-bootstrap-samples
        measure_arr[i] = measure(data[idx_resample])
    ci_l, ci_u = np.percentile(measure_arr, ci_bound)
    return ci_l, ci_u

if __name__  == '__main__':
    import matplotlib.pyplot as plt
    radius = 2
    n_point = 10

    circle_curve = [[radius * np.sin(t), radius * np.cos(t)] for t in np.linspace(0, 2 * np.pi, n_point, endpoint=False)]

    circle_curve = np.array(circle_curve)

    ct, ct_mean = curvature(circle_curve)
    print(ct, ct_mean)
