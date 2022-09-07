import numpy as np

class Mesh_Helper():
    ''' generate mesh and convert mesh (between periodic or nonperiodic) '''
    def __init__(self, var_period):
        self.var_period = var_period

    def set_var_period(self, var_period):
        self.var_period = var_period

    def kernel_to_nonperiodic(self, kernel_width):
        '''
        convert kernel size to nonperiodic. This is same as label_to_nonperiodic, just the data type of kernel_width is a list
        input:
          kernel_width (list [kernel_width_for_info_0, kernel_width_for_info_1, ...])
        output:
          kernel_width same as origin kernel_width of periodic var converted into two seperate kernel_width for cos and sin seperately
        '''
        pass
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

    @staticmethod
    def generate_manifold_label_mesh(mesh_bound, mesh_size, random=None):
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
            if random is None:
                label_mesh[:, i] = np.linspace(mesh_bound[i][0], mesh_bound[i][1], mesh_size)
            else:
                label_mesh[:, i] = np.random.uniform(mesh_bound[i][0], mesh_bound[i][1], mesh_size)
        return label_mesh
