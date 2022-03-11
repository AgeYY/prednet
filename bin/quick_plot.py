import matplotlib.pyplot as plt

e_natural = [ ]
e_synthetic = [1.795, 1.25, 1.42]
#r_natural = [ [0, ] ]
#r_synthetic = [ [0, ] ]

def proc_data(data):
    data_out = [[i, data[i]] for i in len(data)]
    data_out = np.array(data_out)
    return data_out

e_natural = proc_data(e_natural)
e_synthetic = proc_data(e_synthetic)
#r_natural = proc_data(r_natural)
#r_synthetic = proc_data(r_synthetic)

plt.figure()
plt.scatter(e_natural[:, 0], e_natural[:, 1])
plt.scatter(e_synthetic[:, 0], e_synthetic[:, 1])
#plt.scatter(r_natural[:, 0], r_natural[:, 1])
#plt.scatter(r_synthetic[:, 0], r_synthetic[:, 1])
plt.show()

e_natural_pca = [ ]
e_synthetic_pca = [0.935, 0.134, 0.20]

plt.figure()
plt.scatter(e_natural_pca[:, 0], e_natural_pca[:, 1])
plt.scatter(e_synthetic_pca[:, 0], e_synthetic_pca[:, 1])
plt.show()
