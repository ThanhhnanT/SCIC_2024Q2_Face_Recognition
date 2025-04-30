import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))
        squeeze = squeeze.view(squeeze.size(0), -1)  # Flatten
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(-1, self.channels, 1, 1)
        return x * excitation.expand_as(x)
