# test the kernel method for computing the conditional expectation
import numpy as np
import matplotlib.pyplot as plt
import predusion.manifold as mfd
from predusion.static_dataset import Toy_Manifold_Dataset
from predusion.geo_tool import angle_vec

############################## cProfile speed testing
import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
############################## cProfile speed testing

train_size, test_size = 200, 10
label_id = (0, 1)
noise_scale = 0.4

ds = Toy_Manifold_Dataset()
train_label, train_feamap, test_label, test_feamap, test_true_feamap = ds.generate_data(train_size, test_size, noise_scale=noise_scale)
test_vec = ds.tangent_vec(test_label)

def layer_wrapper(*data):
    ''' wrap a null layer for testing the layer_manifold'''
    return [{'null_layer': data_i} for data_i in data]

#### fit the manifold
kernel_width = [0.1, 0.1]

train_feamap_wp, test_feamap_wp = layer_wrapper(train_feamap, test_feamap)

lm = mfd.Layer_manifold()
lm.load_data(train_feamap_wp, train_label)
lm.build_kernel(test_label, label_id=label_id)
lm.fit_info_manifold_all(label_id=label_id)
manifold_fit = lm.ana_group['null_layer'][label_id].info_manifold

### test the fit_info_manifold_grid_all
#x1 = np.linspace(0, 1, 5)
#lm.fit_info_manifold_grid_all([x1, x1])
#manifold_fit = lm.ana_group['null_layer'][(0,1)].info_manifold

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(train_feamap[:, 0], train_feamap[:, 1], train_feamap[:, 2], label='data points', color='grey') # plot out training data
ax.scatter3D(manifold_fit[:, 0], manifold_fit[:, 1], manifold_fit[:, 2], label='gaussian kernel grided label')
ax.scatter3D(test_true_feamap[:, 0], test_true_feamap[:, 1], test_true_feamap[:, 2], label='ground true in testing')
plt.legend()
plt.show()

####### decoding score
score = lm.manifold_decoder_score_all(test_feamap_wp, test_label, label_id=label_id)
print('score: ', score)
dim = lm.dim_all(0.95, label_id)
print('dim: ', dim)

##### calculate tangent vector
vec = lm.tangent_vec_all(label_id)
vec_true = ds.tangent_vec(test_label)
print('estimated tangent vector 0: \n', vec['null_layer'][:3, 0, :])
print('true tangent vector 0: \n', vec_true[:3, 0, :])
print('estimated tangent vector 1: \n', vec['null_layer'][:3, 1, :])
print('true tangent vector 1: \n', vec_true[:3, 1, :])

### calculate angles
ag_est = lm.angle_tangent_vec_all(label_id, label_id)['null_layer']
ag_true = angle_vec(vec_true[:, 0, :], vec_true[:, 1, :])

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(ag_true, ag_est)
dig_line = np.linspace(0, 180, 100)
ax.plot(dig_line, dig_line, '--')
ax.set_xlabel('true angle')
ax.set_ylabel('estimate angle')

############################## cProfile speed testing
pr.disable()
sortby = 'cumtime'
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(5)
############################## cProfile speed testing

plt.show()
