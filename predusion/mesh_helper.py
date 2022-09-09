import numpy as np

class Mesh_Helper():
    ''' generate mesh and convert mesh (between periodic or nonperiodic) '''
    def __init__(self, var_period):
        self.var_period = var_period

    def set_var_period(self, var_period):
        self.var_period = var_period

    def label_id_to_nonperiodic(self, label_id):
        label_id_nonperiodic = []

        nu = [False if vari is None else True for vari in self.var_period] # binary vector to indicate whether the i th variable is periodic or not
        sum_nu = np.cumsum(nu)

        for lid in label_id:
            j = lid + sum_nu[lid]
            if nu[lid]:
                label_id_nonperiodic.append(j - 1)
                label_id_nonperiodic.append(j)
            else:
                label_id_nonperiodic.append(j)

        return tuple(label_id_nonperiodic)

    def label_id_to_origin(self, label_id_nonperiodic):
        label_id = []
        nu = [False if vari is None else True for vari in self.var_period] # binary vector to indicate whether the i th variable is periodic or not
        sum_nu = np.cumsum(nu)

        for i in range(sum_nu.shape[0]):
            j = i + sum_nu[i]
            if j in label_id_nonperiodic:
                label_id.append(i)

        return tuple(label_id)

    def kernel_to_nonperiodic(self, kernel):
        '''
        convert kernel size to nonperiodic. This is same as label_to_nonperiodic, just the data type of kernel_width is a list
        input:
          kernel_width (list [kernel_width_for_info_0, kernel_width_for_info_1, ...])
        output:
          kernel_width same as origin kernel_width of periodic var converted into two seperate kernel_width for cos and sin seperately
        '''
        i_var = 0
        kernel_nonperiodic = []
        for period in self.var_period:
            if period is None:
                kernel_nonperiodic.append(kernel[i_var])
            else:
                T = period[1] - period[0]
                kernel_i_var = 4.0 / np.pi * np.sin(np.pi * kernel[i_var] / T)
                kernel_nonperiodic.append(kernel_i_var)
                kernel_nonperiodic.append(kernel_i_var)
            i_var +=1

        return np.array(kernel_nonperiodic)

    def kernel_to_origin(self, kernel_nonperiodic):
        ''' inverse operation of kernel_to_nonperiod'''
        i_var = 0
        kernel = []
        for period in self.var_period:
            if period is None:
                kernel.append(kernel_nonperiodic[i_var])
            else:
                T = period[1] - period[0]
                sin_val = np.clip(kernel_nonperiodic[i_var] * np.pi / 4.0, -1, 1)
                kernel_i_var = T / np.pi * np.arcsin(sin_val)
                kernel.append(kernel_i_var)
                i_var +=1

            i_var +=1
        return np.array(kernel)

    def label_to_nonperiodic(self, label):
        '''
        var_period (None or list [[a0, b0], [a1, b1], ..., None]): the length is equal to the number of labels (columns of train_label). None means this variable is linear, while given a period interval, this function will convert it into two variable cos and sin
        label (np array, [n_data_points, n_labels])
        '''
        i_var = 0
        label_nonperiod = []
        for period in self.var_period:
            if period is None:
                label_nonperiod.append(label[:, i_var])
            else:
                T = period[1] - period[0]
                angle = (label[:, i_var] - period[0]) / T * 2.0 * np.pi # convert to 0 to 360
                labelcos = np.cos(angle)
                labelsin = np.sin(angle)
                label_nonperiod.append(labelcos)
                label_nonperiod.append(labelsin)
            i_var +=1
        label_nonperiod = np.array(label_nonperiod).transpose()
        return label_nonperiod

    def label_to_origin(self, label_nonperiod):
        '''
        inverse procedure of label_to_nonperiodic
        '''
        i_var = 0
        label = []
        for period in self.var_period:
            if period is None:
                label.append(label_nonperiod[:, i_var])
            else:
                T = period[1] - period[0]
                labelval = np.arctan2(label_nonperiod[:, i_var+1], label_nonperiod[:, i_var])
                label.append( labelval / np.pi * T / 2.0 % T + period[0])
                i_var += 1
            i_var +=1
        label = np.array(label).transpose()
        return label

    def generate_manifold_label_mesh(self, mesh_bound, mesh_size, random=False, grid=False):
        '''
        input:
          mesh_bound (list [[a0, b0], [a1, b1], [a2, b2], ...]): length of list is n_info
          mesh_size (int)
          random (str or None): None means linspace, str could only be uniform in current version
        output:
          label_mesh (np.array [mesh_size, n_info])
        '''
        n_info = len(mesh_bound)
        label_mesh = np.empty((mesh_size, n_info))
        for i in range(n_info):
            if random:
                label_mesh[:, i] = np.random.uniform(mesh_bound[i][0], mesh_bound[i][1], mesh_size)
            else:
                label_mesh[:, i] = np.linspace(mesh_bound[i][0], mesh_bound[i][1], mesh_size)

        if grid:
            label_mesh_list = [label_mesh[:, i] for i in range(n_info)]
            label_mesh = np.array( np.meshgrid(*label_mesh_list)).transpose().reshape( (-1, n_info) )

        return label_mesh

    def generate_kernel_mesh(self, kernel_mesh_bound, kernel_mesh_size, random=None):
        kernel_mesh = self.generate_manifold_label_mesh(kernel_mesh_bound, kernel_mesh_size, random=random)
        return kernel_mesh

    def kernel_mesh_to_nonperiodic(self, kernel_mesh):
        '''This function is useful in validation. Rows are sample, cols are labels'''
        return self.kernel_to_nonperiodic(kernel_mesh.T).T

