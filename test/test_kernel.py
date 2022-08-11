# test the kernel method for computing the conditional expectation
import numpy as np
import matplotlib.pyplot as plt
import predusion.manifold as mfd
import predusion.static_dataset as Toy_Manifold_Dataset

############################## cProfile speed testing
import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
############################## cProfile speed testing

def angle_vec(vec1, vec2, limit_90=False):
    '''
    vec1, vec2: [n_query, n_feature]
    '''
    vec1_u, vec2_u = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True), vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    dot = np.einsum(vec1_u, [0, 1], vec2_u, [0, 1], [0])
    ag = np.arccos(np.clip(dot, -1, 1)) / np.pi * 180
    if limit_90:
        return np.minimum(ag, 180-ag)
    else:
        return ag

train_size, test_size = 1000, 10

ds = Toy_Manifold_Dataset()
train_label, train_feamap, test_label, test_feamap, test_true_feamap = ds.generate_data(train_size, test_size)
test_vec = ds.tangent_vec(test_label)

def layer_wrapper(data):
    ''' wrap a null layer for testing the layer_manifold'''
    return {'null_layer': data}



#### fit the manifold
kernel_width = [0.1, 0.1]

mf = mfd.Data_manifold()
mf.kernel = 'gaussian'
mf.kernel_width = kernel_width
mf.build_kernel(test_label, train_feamap, train_label)
_, manifold_fit = mf.fit_by_label()

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(train_feamap[:, 0], train_feamap[:, 1], train_feamap[:, 2], label='data points', color='grey') # plot out training data
ax.scatter3D(manifold_fit[:, 0], manifold_fit[:, 1], manifold_fit[:, 2], label='gaussian kernel grided label')
ax.scatter3D(test_true_feamap[:, 0], test_true_feamap[:, 1], test_true_feamap[:, 2], label='ground true in testing')
plt.legend()

##### calculate tangent vector
vec = mf.tangent_vector(label_id='all')
vec_true = ds.tangent_vec(test_label)
print('estimated tangent vector 0: \n', vec[:3, 0, :])
print('true tangent vector 0: \n', vec_true[:3, 0, :])
print('estimated tangent vector 1: \n', vec[:3, 1, :])
print('true tangent vector 1: \n', vec_true[:3, 1, :])

### calculate angles
ag_est = angle_vec(vec[:, 0, :], vec[:, 1, :])
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
ps.print_stats(10)
############################## cProfile speed testing

plt.show()
