# test the kernel method for computing the conditional expectation
import numpy as np
import matplotlib.pyplot as plt
import predusion.manifold as manifold

sample_size = 100
mesh_size = 30

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

geo_ana = manifold.Data_manifold()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], label='sample', color='grey')

#_, mean_y = geo_ana.fit_info_manifold([x1_mesh, x2_mesh], y, label, kernel_name='bin', kernel_width=[0.5, 0.5])
#ax.scatter3D(mean_y[:, 0], mean_y[:, 1], mean_y[:, 2], label='bin kernel')

_, mean_y = geo_ana.fit_info_manifold([x1_mesh, x2_mesh], y, label, kernel_name='gaussian')
ax.scatter3D(mean_y[:, 0], mean_y[:, 1], mean_y[:, 2], label='gaussian kernel')

ax.scatter3D(y_mesh_true[:, 0], y_mesh_true[:, 1], y_mesh_true[:, 2], label='ground true')
plt.legend()
plt.show()
