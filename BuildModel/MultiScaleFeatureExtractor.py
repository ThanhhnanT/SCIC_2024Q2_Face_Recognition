import torch
import torch.nn as nn

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_features_list, out_features):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            ) for in_features in in_features_list
        ])

        self.fusion = nn.Sequential(
            nn.Linear(out_features * len(in_features_list), out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_maps):
        features = []
        for i, feature_map in enumerate(feature_maps):
            features.append(self.adapters[i](feature_map).flatten(1))

        concatenated = torch.cat(features, dim=1)
        fused = self.fusion(concatenated)

        return fused