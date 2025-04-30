import torch
import torch.nn as nn

class RegionFeatureExtractor(nn.Module):
    def __init__(self, in_features, feature_dim):
        super(RegionFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_features, feature_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.bn_fc = nn.BatchNorm1d(feature_dim)
        self.relu_fc = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        return x
