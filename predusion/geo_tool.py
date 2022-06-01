import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

def cos_xt_xv(neural_x, error_bar='std'):
    '''
    calculate the mean and se (or std) of one layer. The cos value of two tangent vectors in the same point
    input:
      neural_x ([n_speed, n_time, feature])
    '''
    # calculate the different along the temporal direction
    neural_x_time = np.gradient(neural_x, axis=1)
    neural_x_speed = np.gradient(neural_x, axis=0)

    # collecting all tangent vectors
    neural_x_time_flat = neural_x_time.reshape((-1, neural_x_time.shape[-1]))
    neural_x_speed_flat = neural_x_speed.reshape((-1, neural_x_speed.shape[-1]))
    ## fit the subspace
    #nc = 6
    #pca = PCA(n_components=nc)
    #neural_x_speed_flat = pca.inverse_transform( pca.fit_transform(neural_x_speed_flat) )
    #neural_x_speed = neural_x_speed_flat.reshape( neural_x_speed.shape )
    #print('speed tangent: ', np.cumsum(pca.explained_variance_ratio_))
    #neural_x_time_flat = pca.inverse_transform( pca.fit_transform(neural_x_time_flat) )
    #neural_x_time = neural_x_time_flat.reshape( neural_x_time.shape )
    #print('time tangent: ', np.cumsum(pca.explained_variance_ratio_))

    # calculate the cos
    dot = np.sum(neural_x_time * neural_x_speed, axis=-1) / np.linalg.norm(neural_x_time, axis=2) / np.linalg.norm(neural_x_speed, axis=2)
    #print(np.isnan(dot))

    dot_flat = dot.flatten()

    if error_bar=='std':
        err = np.std(dot_flat)
    else: # sem
        err = np.std(dot_flat) / np.sqrt(np.size(dot_flat))

    return np.mean(dot_flat), err

def cos_para_layer(neural_x, error_bar='std'):
    '''
    calculate the mean and se (or std) of one layer
    input:
      neural_x ([n_speed, n_time, feature])

    '''
    # calculate the different along the temporal direction
    #neural_x_speed = np.gradient(neural_x, axis=0)
    neural_x_speed_mean = np.mean(neural_x, axis=0)
    #shift_neural_x = np.tile(np.expand_dims(neural_x[0], axis=0), (cut, 1, 1))
    shift_neural_x = np.tile(np.expand_dims(neural_x_speed_mean, axis=0), (cut, 1 , 1))
    #neural_x_speed = neural_x - np.tile(neural_x[:, 0])
    neural_x_speed = neural_x - shift_neural_x
    #neural_x_speed = neural_x_speed[1:, :]

    nt = neural_x_speed.shape[1]
    # calculate the cos
    dot = []

    for i in range(nt - 1):
        for j in range(i + 1, nt):
            dot.append( np.sum(neural_x_speed[:, i] * neural_x_speed[:, j], axis=-1) / np.linalg.norm(neural_x_speed[:, i], axis=-1) / np.linalg.norm(neural_x_speed[:, j], axis=-1) )
            #dot.append(frdist(neural_x_speed[:, i], neural_x_speed[:, j]))

    dot_flat = np.array(dot).flatten()

    if error_bar=='std':
        err = np.std(dot_flat)
    else: # sem
        err = np.std(dot_flat) / np.sqrt(np.size(dot_flat))

    return np.mean(dot_flat), err

def pca_reduce(neural_x, n_components=None, print_message=False, with_var_explained=False):
    '''
    reduce the dimensionality using pca
    input:
      neural_x ([n_speed, n_time, feature])
    output:
      neural_x ([n_speed, n_time, n_components])
      total_var_explained (float)
    '''
    if n_components is None:
        return neural_x

    # pca processing
    neural_x_flat = neural_x.reshape(-1, neural_x.shape[-1])
    pca = PCA(n_components=n_components)
    neural_x_flat = pca.fit_transform(neural_x_flat)
    neural_x = neural_x_flat.reshape(neural_x.shape[0], neural_x.shape[1], n_components)
    if print_message:
        print('the total variance explained is: ', np.cumsum(pca.explained_variance_ratio_)[-1])
    if with_var_explained:
        return neural_x, np.cumsum(pca.explained_variance_ratio_)

    return neural_x

def trans_norm(curve):
    mean = np.mean(curve, axis=0)
    mean = np.tile(mean[np.newaxis, ...], (curve.shape[0], 1))
    var = np.sum(np.var(curve, axis=0))
    curve_transformed = (curve - mean) / np.sqrt(var)
    return curve_transformed

def procrustes_translate_normalize(curve1, curve2):
    '''
    curve ([n_points, n_dim])
    '''
    curve1_t = trans_norm(curve1)
    curve2_t = trans_norm(curve2)
    d = (curve1_t - curve2_t)**2
    return np.sum(d.flatten())

def procrustes_curve_diff_time(neural_x, error_bar='std', n_com=3, print_message=False, method=None):
    '''
    calculate the disparity between curves x(t1, v) and x(t2, v)
    input:
      neural_x ([n_speed, n_time, feature])
    '''

    disparity = []

    for i in range(neural_x.shape[1] - 1):
        for j in range(i + 1, neural_x.shape[1]):
            neural_x_pair = neural_x[:, [i, j], :]
            neural_x_pair = pca_reduce(neural_x_pair, n_components=n_com, print_message=print_message)
            if method == 'tran_norm':
                disparity_temp = procrustes_translate_normalize(neural_x_pair[:, 0], neural_x_pair[:, 1])

            else:
                _, _, disparity_temp = procrustes(neural_x_pair[:, 0], neural_x_pair[:, 1])
            disparity.append(disparity_temp)

    if error_bar=='std':
        err = np.std(disparity)
    else: # sem
        err = np.std(disparity) / np.sqrt(np.size(disparity))

    return np.mean(disparity), err

def dim_manifold(neural_x, error_bar='std', n_com_max=None, print_message=False, thresh=0.95):
    '''
    dimensionality of the information manifold
    '''
    if n_com_max is None:
        n_com_max = neural_x.shape[0]
    pca = PCA(n_components=n_com_max)
    dim = []
    for i in range(neural_x.shape[1]):
        pca.fit(neural_x[:, i])
        var_explained = np.cumsum(pca.explained_variance_ratio_)

        for j, var_exp in enumerate(var_explained):
            if var_exp > thresh: break
        dim.append(j)

    if error_bar=='std':
        err = np.std(dim)
    else: # sem
        err = np.std(dim) / np.sqrt(np.size(dim))

    return np.mean(dim), err

def ratio_speed_time(neural_x, error_bar='std', n_com=None, print_message=False):
    '''
    calculate the mean and se (or std) of one layer
    input:
      neural_x ([n_speed, n_time, feature])

    '''
    var_time, var_speed = [], []
    # variance of time
    for i_sp in range(neural_x.shape[0]):
        neural_x_time = neural_x[i_sp]
        var_time_temp = np.sum(np.var(neural_x_time, axis=0))
        var_time.append(var_time_temp)
    for i_t in range(neural_x.shape[1]):
        neural_x_speed = neural_x[:, i_t]
        var_speed_temp = np.sum(np.var(neural_x_speed, axis=0))
        var_speed.append(var_speed_temp)
    std_t, std_s = np.std(var_time), np.std(var_speed)
    mean_var_t, mean_var_s = np.mean(var_time), np.mean(var_speed)
    ratio = mean_var_s / mean_var_t
    std_ratio = ratio * np.sqrt((std_s / mean_var_s)**2 + (std_t / mean_var_t)**2) # propagation of error

    return ratio, std_ratio
