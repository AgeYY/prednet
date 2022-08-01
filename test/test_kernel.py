# test the kernel method for computing the conditional expectation
import numpy as np
import matplotlib.pyplot as plt
import predusion.geo_tool as geo_tool

sample_size = 100
mesh_size = 30

#def func(x):
#    size = x.shape[0]
#    y1 = x
#    y2 = 3 * x
#    return np.array([y1, y2]).T
#
#def add_func_noise(y, scale=0.5):
#    return y + np.random.normal(0, scale, y.shape)
#
#x = np.random.normal(0, 1, sample_size) # this is label
#y = func(x) # this is is the feature maps
#y = add_func_noise(y)
#
#x_mesh = np.linspace(-1, 1, mesh_size)
#y_mesh_true = func(x_mesh)
#
#geo_ana = geo_tool.Single_geo_analyzer()
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(x, y[:, 0], y[:, 1], label='sample', color='grey')
#
#_, mean_y = geo_ana.fit_info_manifold(x_mesh, y, x, kernel_name='bin')
#ax.plot(x_mesh, mean_y[:, 0], mean_y[:, 1], label='bin kernel')
#
#_, mean_y = geo_ana.fit_info_manifold(x_mesh, y, x, kernel_name='gaussian')
#ax.plot(x_mesh, mean_y[:, 0], mean_y[:, 1], label='gaussian kernel')
#
#ax.plot(x_mesh, y_mesh_true[:, 0], y_mesh_true[:, 1], label='ground true')
#plt.legend()
#plt.show()

#### test double geo analyzer
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

x1 = np.random.normal(0, 1, sample_size) # this is label
x2 = np.random.normal(0, 1, sample_size) # this is label
label = np.array([x1, x2]).T

y = func(x1, x2) # this is is the feature maps
y = add_func_noise(y)

x1_mesh = np.random.normal(0, 1, mesh_size) # this is label
x2_mesh = np.random.normal(0, 1, mesh_size) # this is label
#x1_mesh = np.linspace(-1, 1, mesh_size) # this is label
#x2_mesh = np.linspace(-1, 1, mesh_size) # this is label
y_mesh_true = func(x1_mesh, x2_mesh)

geo_ana = geo_tool.Double_geo_analyzer()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], label='sample', color='grey')

_, mean_y = geo_ana.fit_info_manifold([x1_mesh, x2_mesh], y, label, kernel_name='bin')
ax.scatter3D(mean_y[:, 0], mean_y[:, 1], mean_y[:, 2], label='bin kernel')

_, mean_y = geo_ana.fit_info_manifold([x1_mesh, x2_mesh], y, label, kernel_name='gaussian')
ax.scatter3D(mean_y[:, 0], mean_y[:, 1], mean_y[:, 2], label='gaussian kernel')

ax.scatter3D(y_mesh_true[:, 0], y_mesh_true[:, 1], y_mesh_true[:, 2], label='ground true')
plt.legend()
plt.show()
