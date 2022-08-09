# test the kernel method for computing the conditional expectation
import numpy as np
import matplotlib.pyplot as plt
import predusion.manifold as mfd

############################## cProfile speed testing
import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
############################## cProfile speed testing

sample_size = 2000
mesh_size = 2

#### generating feamap and label
def func(x1, x2):
    '''
    x ([n_observation, n_independent_variable])
    '''
    y1 = x1
    y2 = 3 * x2
    y3 = x1*x1 + x2*x2
    return np.array([y1, y2, y3]).T

def add_func_noise(y, scale=0.4):
    return y + np.random.normal(0, scale, y.shape)

def tangent_vec(x1, x2, label_id=0):
    n_sample = x1.shape
    if label_id == 0:
        y1 = np.ones(n_sample)
        y2 = np.zeros(n_sample)
        y3 = 2 * x1
    elif label_id == 1:
        y1 = np.zeros(n_sample)
        y2 = 3 * np.ones(n_sample)
        y3 = 2 * x2
    return np.array([y1, y2, y3]).T

x1 = np.random.normal(0, 1, sample_size) # this is label
x2 = np.random.normal(0, 1, sample_size) # this is label
label = np.array([x1, x2]).T

y = func(x1, x2) # this is is the feature maps
feamap = add_func_noise(y)

#### generate mesh points
x1_mesh = np.random.normal(0, 1, mesh_size) # this is label
x2_mesh = np.random.normal(0, 1, mesh_size) # this is label

#### calculate the ground turth for these mesh points
x1_mesh_4_ground, x2_mesh_4_ground = np.meshgrid(x1_mesh, x2_mesh)
manifold_mesh_true = func(x1_mesh_4_ground.flatten(), x2_mesh_4_ground.flatten())

#### fit the manifold
kernel_width = [0.1, 0.1]

geo_ana = mfd.Data_manifold()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(feamap[:, 0], feamap[:, 1], feamap[:, 2], label='data points', color='grey')

geo_ana.kernel = 'gaussian'
geo_ana.kernel_width = kernel_width
_, manifold_fit = geo_ana.fit_by_label_grid_mesh([x1_mesh, x2_mesh], feamap, label)
ax.scatter3D(manifold_fit[:, 0], manifold_fit[:, 1], manifold_fit[:, 2], label='gaussian kernel ungrided label')

label_mesh = np.array( np.meshgrid(x1_mesh, x2_mesh) ).transpose().reshape(-1, 2) # this is used for testing fit_by_label
geo_ana.build_kernel(label_mesh, feamap, label)
_, manifold_fit = geo_ana.fit_by_label()
ax.scatter3D(manifold_fit[:, 0], manifold_fit[:, 1], manifold_fit[:, 2], label='gaussian kernel grided label')

#### calculate tangent vector
vec = geo_ana.tangent_vector(label_id=0)
vec_true = tangent_vec(x1_mesh_4_ground, x2_mesh_4_ground, label_id=0)
print(vec)
print(vec_true)

############################## cProfile speed testing
pr.disable()
sortby = 'cumtime'
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(10)
############################## cProfile speed testing

ax.scatter3D(manifold_mesh_true[:, 0], manifold_mesh_true[:, 1], manifold_mesh_true[:, 2], label='ground true')
plt.legend()
plt.show()
