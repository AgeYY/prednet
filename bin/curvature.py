# calculate the curverture
import numpy as np
import matplotlib.pyplot as plt

from predusion.tools import curvature


radius = 2
n_point = 10

circle_curve = [[radius * np.sin(t), radius * np.cos(t)] for t in np.linspace(0, 2 * np.pi, n_point, endpoint=False)]

circle_curve = np.array(circle_curve)

#plt.figure()
#plt.scatter(circle_curve[:, 0], circle_curve[:, 1])
#plt.show()

ct, ct_mean = curvature(circle_curve)
print(ct, ct_mean)
