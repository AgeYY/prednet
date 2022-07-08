# generate the neural response, input data and the label of MLP
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from predusion.ploter import plot_dimension_reduction, Ploter_dim_reduction
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from predusion.mlp_agent import MLP
from predusion.mlp_trainer import train, evaluate
from predusion.static_dataset import Surface_dataset
from predusion.geo_tool import angle_PC, angle_between
from predusion.pls_analyzer import PLS_pair
from kitti_settings import *


INPUT_DIM = 3
OUTPUT_DIM = 2
n_layers = 4 # input, layer 1, ... exclude the output prediction only
tot_title = 'Untrained MLP'
#surface_shape = 'cylinder'
surface_shape = 'helicoid'

r = 1
helicoid_alpha, helicoid_beta = 1, 2

weight_decay = 0.001
train_ratio = 0.8
n_epoch = 20
n_sample = 200
model_path = os.path.join(WEIGHTS_DIR, 'mlp_cylinder.pt')  # where weights will be saved
n_component_visu = 2

lt1 = np.random.uniform(0, 2 * np.pi, n_sample)
lt2 = np.random.uniform(-4, 4, n_sample)
label = np.array([lt1, lt2]).T
length = n_sample

dataset = Surface_dataset(length=length, to_torch=True, shape=surface_shape, label=label)
train_len = int(len(dataset) * train_ratio)
test_len = len(dataset) - train_len
train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

train_iterator = DataLoader(train_set, batch_size=128, shuffle=True)
test_iterator = DataLoader(test_set, batch_size=128, shuffle=True)

# train the MLP
model = MLP(INPUT_DIM, OUTPUT_DIM)

optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
criterion = nn.MSELoss()
#criterion = loss_cylinder

for i in range(n_epoch):
    train_loss = train(model, train_iterator, optimizer, criterion)
    eval_loss= evaluate(model, test_iterator, criterion)
    torch.save(model.state_dict(), model_path)

    if i % 50 == 0:
        print('eval loss: ', eval_loss)
        print('train loss: ', train_loss)

# visualize the feature map
feamap = model.feature_map(dataset[:]['X'])
for key in feamap:
    feamap[key] = feamap[key].cpu().detach().numpy()
label = dataset[:]['label'].cpu().detach().numpy()

hkl.dump(label, './kitti_data/label_' + 'surface_shape_mlp.hkl')
hkl.dump(feamap, './kitti_data/neural_' + 'surface_shape_mlp.hkl')
