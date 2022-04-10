import numpy as np
from predusion.tools import confidence_interval

class Curvature():
    def __init__(self, data=None):
        '''
        data ([n_video, n_video_frames, n_neurons]): feature map to a video
        '''
        self.data = data

    def load_data(self, data):
        self.data = data

    def curvature_frame(self, n_component=None):
        '''
        calculate the curvatures of the data
        '''
        ct = []
        for i, video in enumerate(self.data):
            if n_component is None:
                ct_temp, _ = curvature(video)
            else:
                ct_temp, _, _ = curvature_pca(video, n_component=n_component)
            ct.append(ct_temp)

        return np.array(ct)

    def curvature_traj(self, cutoff=0, n_component=None):
        '''
        cutoff (int): the curvature of a trajactory is the average of curvatures from cutoff frame to the end of video
        ct ([n_video])
        '''
        ct_temp = self.curvature_frame(n_component=n_component)
        ct = np.mean(ct_temp[:, cutoff:], axis=1)
        return ct

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

from sklearn.decomposition import PCA
def curvature_pca(data, n_component=2):
    '''
    data ([n_video_frames, n_neurons]): feature map to a video
    '''
    pca = PCA(n_components=n_component)
    data_pca = pca.fit_transform(data)

    ct, ct_mean = curvature(data_pca)

    return ct, ct_mean, pca.explained_variance_ratio_

def curv_video_neural(video_ppd, sub, output_mode, cutoff=2, n_component=5, ci_bound=[16.5, 83.5], ci_resample=1000):
    '''
    input:
      video_ppd ([n_video, n_frame, *imshape, 3 channels]): processed video
      cutoff (int, smaller than n_frame): the curvature of a trajectory is the mean from curvature from the cutoff frame to the end. Due to the cutoff, the curvature of artificial video is no longer the same as natural video, but the affect should be minor
      sub (prednet agent)
      output_mode (string list): a list of output modes, e.g. ['E0', 'R0']
      n_component (int): number of principal compoents
    output:
    output the change of curvature for oringinal video and neural response in different layers.
      ct_mean_change_median (dict): {'E0': -0.13, 'R0': 0.1}, key means layer, number means change of curvature without pca, median of all videos
      ct_mean_change_ci (dict): ci bound
      ct_mean_change_pca_median (dict): {'E0': -0.13, 'R0': 0.1}, key means layer, number means change of curvature without pca, median of all videos. After pca
      ct_mean_change_pca_ci (dict)
    '''
    cv = Curvature()
    # calculate the curvature of video
    video_flat = video_ppd.reshape((video_ppd.shape[0], video_ppd.shape[1], -1))
    cv.load_data(video_flat)
    ct_mean_video = cv.curvature_traj(cutoff=cutoff)

    ct_mean_video_pca = cv.curvature_traj(cutoff=cutoff, n_component=n_component)

    ##### load the prednet
    batch_size = video_ppd.shape[0]

    output = sub.output_multiple(video_ppd, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)

    ########## curverature of the neural response
    ct_mean = {} # curvature without pca
    ct_mean_pca = {} # curvature with pca

    for key in output:
        output[key] = output[key].reshape(output[key].shape[0], output[key].shape[1], -1) # flatten (n_video, n_frames, n_neurons)
        cv.load_data(output[key])

        ct_mean[key] = cv.curvature_traj(cutoff=cutoff, n_component=None) # ignore the first two video frames due to the pool prediction of the prednet
        ct_mean_pca[key] = cv.curvature_traj(cutoff=cutoff, n_component=n_component) # ignore the first two video frames due to the pool prediction of the prednet

    ########## Change of curvature
    ct_mean_change = {}
    ct_mean_change_pca = {}

    ct_mean_change_median = {}
    ct_mean_change_pca_median = {}

    ct_mean_change_ci = {}
    ct_mean_change_pca_ci = {}

    for key in ct_mean:
        ct_mean_change[key] = ct_mean[key] - ct_mean_video
        ct_mean_change_median[key] = np.median(ct_mean_change[key])
        ct_mean_change_ci[key] = np.array( confidence_interval(ct_mean_change[key], ci_bound=ci_bound, measure=np.median, n_resamples=ci_resample) )
        
        ct_mean_change_pca[key] = ct_mean_pca[key] - ct_mean_video_pca
        ct_mean_change_pca_median[key] = np.median(ct_mean_change_pca[key])
        ct_mean_change_pca_ci[key] = np.array( confidence_interval(ct_mean_change_pca[key], ci_bound=ci_bound, measure=np.median, n_resamples=ci_resample) )

    return ct_mean_change_median, ct_mean_change_ci, ct_mean_change_pca_median, ct_mean_change_pca_ci
