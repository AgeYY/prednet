# mlp
import torch.nn as nn
import torch.nn.functional as F

## train MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hfc1 = nn.Linear(250, 100)
        self.hfc2 = nn.Linear(100, 50)
        self.output_fc = nn.Linear(50, output_dim)

    def forward(self, x):

        # x = [batch size, features...]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = [batch size, feratures]
        h1 = F.relu(self.input_fc(x))
        # h1 = [batch size, 250]
        h2 = F.relu(self.hfc1(h1))
        # h2 = [batch size, 100]
        h3 = F.relu(self.hfc2(h2))
        y_pred = self.output_fc(h3)
        # y_pred = [batch size, output dim]
        return y_pred

    def feature_map(self, x, include_input=True):
        # x = [batch size, features...]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = [batch size, feratures]
        h1 = F.relu(self.input_fc(x))
        # h1 = [batch size, 250]
        h2 = F.relu(self.hfc1(h1))
        # h2 = [batch size, 100]
        h3 = F.relu(self.hfc2(h2))
        y_pred = self.output_fc(h3)

        fea_map = {'y_pred': y_pred, 'h3': h3, 'h2': h2, 'h1': h1}

        if include_input:
            fea_map['X'] = x

        return fea_map
