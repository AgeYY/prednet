import torch
import numpy as np
import torch.nn as nn

#def loss_cylinder(output, target):
#    '''
#    output (tensor, [n_sample, 2]): first columns is theta, second one is z.
#    reminder is not a differential function, training may be failed
#    '''
#    output_per = output.clone()
#    output_per[:, 0] = torch.remainder(output[:, 0], 2 * np.pi)
#    return nn.functional.mse_loss(output_per, target)

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for sample in iterator:

        x = sample['X']
        y = torch.stack((sample['theta'], sample['z']))
        y = torch.transpose( y, 0, 1)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

