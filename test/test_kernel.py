# test the kernel method for computing the conditional expectation
import numpy as np
import matplotlib.pyplot as plt
import predusion.geo_tool as geo_tool

sample_size = 50
mesh_size = 100

def func(x):
    size = x.shape[0]
    y1 = x
    y2 = 3 * x
    return np.array([y1, y2]).T

def add_func_noise(y, scale=0.5):
    return y + np.random.normal(0, scale, y.shape)

x = np.random.normal(0, 1, sample_size) # this is label
y = func(x) # this is is the feature maps
y = add_func_noise(y)

x_mesh = np.linspace(-1, 1, mesh_size)
y_mesh_true = func(x_mesh)

geo_ana = geo_tool.Single_geo_analyzer()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y[:, 0], y[:, 1], label='sample', color='grey')

_, mean_y = geo_ana.fit_info_manifold(x_mesh, y, x, kernel='bin')
ax.plot(x_mesh, mean_y[:, 0], mean_y[:, 1], label='bin kernel')

_, mean_y = geo_ana.fit_info_manifold(x_mesh, y, x, kernel='gaussian')
ax.plot(x_mesh, mean_y[:, 0], mean_y[:, 1], label='gaussian kernel')

ax.plot(x_mesh, y_mesh_true[:, 0], y_mesh_true[:, 1], label='ground true')
plt.legend()
plt.show()


