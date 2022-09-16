######################################## For drifting grating prednet
out_data_head = 'grating_stim'
output_mode = ['R0', 'R1', 'R2', 'R3']
neural_data_path = 'neural_' + out_data_head + '_R_prednet' + '.hkl'
label_path = 'label_' + out_data_head + '_R_prednet' + '.hkl'
label_name_path = 'label_name_' + out_data_head + '_R_prednet' + '.hkl'

train_ratio = 0.6
test_ratio = 0.2
explained_var_thre = 0.90
explained_var_thre_pca_all_data = 0.90
# drifting grating configurations
label_id = (0,1)

mesh_size = 30
mesh_bound = [[0, 0.12], [0, 360], [0, 5]]
mesh_random = False
mesh_grid = True # grid mesh makes better visualization

kernel_mesh_size = 5
kernel_mesh_bound = [[0.02, 0.05], [20, 40], [0.01, 1]]
## step 1: carse search using grid, single label_id
#kernel_random = False
## step 2: refine kernel_mesh_bound acording to step 1, use random search
kernel_mesh_bound = [[0.02, 0.05], [20, 40], [0.01, 1]]
kernel_random = True

var_period = [None, [0, 360], None] # the length is equal to the number of labels (columns of train_label). None means this variable is linear, while given a period interval, von mises function would be used as a kernel
kernel_width = [0.033, 37, 1]
n_layer = 5
layer_order = ['X', 'R0', 'R1', 'R2', 'R3']

######################################## For drifting grating prednet

