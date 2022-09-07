import numpy as np
from predusion.static_dataset import Layer_Dataset

def label_to_nonperiodic(label, var_period):
    '''
    var_period (None or list [[a0, b0], [a1, b1], ..., None]): the length is equal to the number of labels (columns of train_label). None means this variable is linear, while given a period interval, von mises function would be used as a kernel
    label (np array, [n_data_points, n_labels])
    '''
    i_var = 0
    label_nonperiod = []
    for period in var_period:
        if period is None:
            label_nonperiod.append(label[:, i_var])
        else:
            T = period[1] - period[0]
            angle = (label[:, i_var] - period[0]) / T * 2.0 * np.pi
            labelcos = np.cos(angle)
            labelsin = np.sin(angle)
            label_nonperiod.append(labelcos)
            label_nonperiod.append(labelsin)
        i_var +=1
    label_nonperiod = np.array(label_nonperiod).transpose()
    return label_nonperiod

def label_to_origin(label_nonperiod, var_period):
    '''
    inverse procedure of label_to_nonperiodic
    '''
    i_var = 0
    label = []
    for period in var_period:
        if period is None:
            label.append(label_nonperiod[:, i_var])
        else:
            center = np.mean(period)
            T = period[1] - period[0]
            labelval = np.arctan2(label_nonperiod[:, i_var+1], label_nonperiod[:, i_var])
            label.append( labelval / np.pi * T / 2.0 % T + period[0])
            i_var += 1
        i_var +=1
    label = np.array(label).transpose()
    return label

label = np.random.uniform(0, 360, size=(10, 3))
var_period = [None, [0, 360], None]

label_nonperiodic = label_to_nonperiodic(label, var_period)

print(label_nonperiodic)

label_recover = label_to_origin(label_nonperiodic, var_period)

print(label_recover, '\n', label)


