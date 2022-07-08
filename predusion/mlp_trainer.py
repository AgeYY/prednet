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
    return sample[x_key], sample[y_key]

def train(model, iterator, optimizer, criterion, x_key='X', y_key='label'):
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

def evaluate(model, iterator, criterion, x_key='X', y_key='label'):

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
