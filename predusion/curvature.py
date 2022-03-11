import numpy as np

class Curvature():
    def __init__(self, data=None):
        '''
        data ([n_video_frames, n_neurons]): feature map to a video
        '''
        self.data = data

    def load_data(self, data):
        self.data = data

    def curvature_frame(self, n_component=None):
        '''
        calculate the curvatures of the data
        ct ([n_video, n_frames])
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
