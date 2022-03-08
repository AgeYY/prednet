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

def curvature(traj):
    '''
    calculate the curvature of a trajectory. Refer more detail to the method section of Hénaff et al. (2021) https://doi.org/10.1038/s41593-019-0377-4
    input:
      traj ([n_point, n_dim])
    output:
      ct ([n_point]): curvature of two adjacent points
      ct_mean: averaged curvature across all points
    '''
    vt_diff = np.diff(traj, axis=0)

    norm = np.linalg.norm(vt_diff, axis=1)

    for t in np.arange(vt_diff.shape[0]):
        vt_diff[t, :] = vt_diff[t, :] / norm[t]

    ct = np.zeros(vt_diff.shape[0] - 1)
    for t in range(ct.shape[0]):
        ct[t] = np.arccos(np.dot(vt_diff[t], vt_diff[t + 1]))

    return ct, np.mean(ct)

if __name__  == '__main__':
    import matplotlib.pyplot as plt
    radius = 2
    n_point = 10

    circle_curve = [[radius * np.sin(t), radius * np.cos(t)] for t in np.linspace(0, 2 * np.pi, n_point, endpoint=False)]

    circle_curve = np.array(circle_curve)

    ct, ct_mean = curvature(circle_curve)
    print(ct, ct_mean)
