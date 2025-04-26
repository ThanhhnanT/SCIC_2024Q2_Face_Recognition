import torch
import torch.nn as nn
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, feature_dim=512, num_classes=1000, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(embeddings), torch.nn.functional.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = self.s * (one_hot * target_logits + (1.0 - one_hot) * cosine)
        return output
