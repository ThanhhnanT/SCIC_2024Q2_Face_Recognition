import torch
import torch.nn as nn
import math
from Dataset import FaceDataset



class SubCenterArcFaceLoss(nn.Module):
    def __init__(self, feature_dim=512, num_classes=6857, s=50, m=0.50, k=3):
        super().__init__()
        self.s = s
        self.m = m
        self.k = k  # Number of sub-centers per identity

        # Create k sub-centers for each class
        # Shape: [num_classes, k, feature_dim]
        self.weight = nn.Parameter(torch.randn(num_classes, k, feature_dim))
        nn.init.xavier_uniform_(self.weight.view(-1, feature_dim))

    def forward(self, embeddings, labels):
        # Move weights to the same device and dtype as embeddings
        weight = self.weight.to(embeddings.device, embeddings.dtype)
        batch_size = embeddings.size(0)

        # Normalize embeddings
        normalized_embeddings = torch.nn.functional.normalize(embeddings)  # [batch_size, feature_dim]

        # Reshape and normalize weights
        weight_view = weight.view(-1, weight.size(-1))  # [num_classes*k, feature_dim]
        normalized_weights = torch.nn.functional.normalize(weight_view)
        normalized_weights = normalized_weights.view(weight.size())  # [num_classes, k, feature_dim]

        # Calculate cosine similarity for all sub-centers of all classes
        # Expand embeddings to compare with each sub-center
        expanded_embeddings = normalized_embeddings.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, feature_dim]
        expanded_weights = normalized_weights.unsqueeze(0)  # [1, num_classes, k, feature_dim]

        # Calculate cosine similarity between embeddings and all sub-centers
        # Result shape: [batch_size, num_classes, k]
        cosine = torch.sum(expanded_embeddings * expanded_weights, dim=-1)

        # For each class, get the maximum similarity among all its sub-centers
        # Shape: [batch_size, num_classes]
        cosine_max, _ = torch.max(cosine, dim=2)

        # Clamp and calculate theta
        theta = torch.acos(torch.clamp(cosine_max, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine_max)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin and scaling
        output = self.s * (one_hot * target_logits + (1.0 - one_hot) * cosine_max)

        return output

    def get_best_subcenter_idx(self, embeddings, labels):
        """
        Returns the index of the best sub-center for each embedding.
        Useful for visualization or analysis.
        """
        weight = self.weight.to(embeddings.device, embeddings.dtype)
        batch_size = embeddings.size(0)

        # Normalize embeddings
        normalized_embeddings = torch.nn.functional.normalize(embeddings)

        # Get the weights for the labels
        selected_weights = weight[labels]  # [batch_size, k, feature_dim]
        normalized_selected_weights = torch.nn.functional.normalize(selected_weights, dim=2)

        # Calculate cosine similarity between embeddings and corresponding sub-centers
        expanded_embeddings = normalized_embeddings.unsqueeze(1)  # [batch_size, 1, feature_dim]
        cosine = torch.sum(expanded_embeddings * normalized_selected_weights, dim=2)  # [batch_size, k]

        # Get the index of the maximum similarity
        _, best_subcenter_idx = torch.max(cosine, dim=1)  # [batch_size]

        return best_subcenter_idx