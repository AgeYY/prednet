import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

#def loss_cylinder(output, target):
#    '''
#    output (tensor, [n_sample, 2]): first columns is theta, second one is z.
#    reminder is not a differential function, training may be failed
#    '''
#    output_per = output.clone()
#    output_per[:, 0] = torch.remainder(output[:, 0], 2 * np.pi)
#    return nn.functional.mse_loss(output_per, target)


def sample_2_inout(sample, x_key, y_key):
    '''
    convert sample into the standard X, Y for training. sample is a dict decribed in def train.
    '''
    x = sample[x_key]
    y = []
    for key in y_key:
        y.append(sample[key])
    y = torch.transpose(torch.stack(y), 0, 1)
    return x, y

def train(model, iterator, optimizer, criterion, x_key='X', y_key=['y']):
    '''
    each sample in the iterator is a dict.
    y_key (list of strings): [y_key1, y_key2, ...]
    sample[x_key] (torch tensor [num_observations, num_features]): input
    sample[y_key1] (torch tensor [num_observations]): label1. all labels would be stacked into a y matrix with shape [num_observations, num_labels (i.e. number of keys)]
    '''

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for sample in tqdm(iterator, desc="Training", leave=False):

        x, y = sample_2_inout(sample, x_key, y_key)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, x_key='X', y_key=['y']):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for sample in tqdm(iterator, desc="Evaluating", leave=False):

            x, y = sample_2_inout(sample, x_key, y_key)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
