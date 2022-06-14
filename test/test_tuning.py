import matplotlib.pyplot as plt
import numpy as np

# orientation tuning
def fs(s, s_max=0, sigma=1):
    return np.exp(-1.0/2.0*(  (s  - s_max) / sigma )**2)

s = np.linspace(-10, 10, 200)

s_max = np.array([0, 2, 4])

def make_figure(sigma):
    y = []
    for i, sm in enumerate(s_max):
        y.append(fs(s, s_max=sm, sigma=sigma))

    return y

sigma = np.linspace(0.1, 2, 20)
for sig in sigma:
    y = make_figure(sig)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(y[0], y[1], y[2])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
