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

def pca_reduce(neural_x, n_components=20, print_message=False, with_var_explained=False):
    '''
    reduce the dimensionality using pca
    input:
      neural_x ([n_speed, n_time, feature])
    output:
      neural_x ([n_speed, n_time, n_components])
      total_var_explained (float)
    '''
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

def procrustes_curve_diff_time(neural_x, error_bar='std', n_com=3, print_message=False):
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
