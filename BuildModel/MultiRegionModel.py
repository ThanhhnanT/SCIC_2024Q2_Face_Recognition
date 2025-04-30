import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.cuda.amp import autocast
from BuildModel.CBAM import CBAM
from BuildModel.SEBlock import SEBlock
from BuildModel.CoordinateAttention import CoordinateAttention
from BuildModel.RegionFeatureExtractor import RegionFeatureExtractor
from BuildModel.MultiScaleFeatureExtractor import  MultiScaleFeatureExtractor


class TransformerFeatureAggregator(nn.Module):
    def __init__(self, feature_dim, num_heads=4, num_layers=2):
        super(TransformerFeatureAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        # features shape: (batch_size, num_regions, feature_dim)
        transformed = self.transformer(features)
        # Average pooling over regions
        pooled = torch.mean(transformed, dim=1)
        return self.fusion(pooled)


class EnhancedMultiRegionModel(nn.Module):
    def __init__(self, feature_dim=512, num_regions=3):
        super(EnhancedMultiRegionModel, self).__init__()

        # Use EfficientNetV2-S as backbone
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Keep track of intermediate feature maps
        self.feature_maps = []

        # Get feature dimensions for EfficientNetV2-S
        # The output channels of each block in EfficientNetV2-S
        self.feature_map_channels = [24, 48, 64, 128, 160, 256, 1280]

        # Hook to capture intermediate feature maps
        def get_activation(block_idx):
            def hook(model, input, output):
                self.feature_maps[block_idx] = output

            return hook

        # Register hooks for intermediate features
        self.feature_maps = [None] * 7

        # Register hooks for the blocks we want to extract features from
        self.backbone.features[1].register_forward_hook(get_activation(0))  # First block
        self.backbone.features[2].register_forward_hook(get_activation(1))  # Second block
        self.backbone.features[3].register_forward_hook(get_activation(2))  # Third block
        self.backbone.features[4].register_forward_hook(get_activation(3))  # Fourth block
        self.backbone.features[5].register_forward_hook(get_activation(4))  # Fifth block
        self.backbone.features[6].register_forward_hook(get_activation(5))  # Sixth block
        self.backbone.features[7].register_forward_hook(get_activation(6))  # Seventh block (final features)

        # Remove the classifier
        self.backbone.classifier = nn.Identity()

        # Add attention modules to different levels
        self.cbam_mid = CBAM(self.feature_map_channels[3])  # Mid-level features
        self.cbam_high = CBAM(self.feature_map_channels[6])  # High-level features

        self.se_blocks = nn.ModuleList([
            SEBlock(channels) for channels in self.feature_map_channels[4:7]  # Higher level features
        ])

        self.coord_attention = CoordinateAttention(self.feature_map_channels[6])

        # Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(
            [self.feature_map_channels[3], self.feature_map_channels[5], self.feature_map_channels[6]],
            feature_dim
        )

        # Region-specific feature extractors
        self.region_extractors = nn.ModuleList([
            RegionFeatureExtractor(self.feature_map_channels[6], feature_dim)
            for _ in range(num_regions)
        ])

        # Region classification
        self.region_classifier = nn.Sequential(
            nn.Conv2d(self.feature_map_channels[6], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_regions)
        )

        # Feature aggregation using transformer
        self.feature_aggregator = TransformerFeatureAggregator(
            feature_dim=feature_dim,
            num_heads=4,
            num_layers=2
        )

        # Final classifier
        self.final = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        with autocast():  # Enable mixed precision
            # Forward through backbone to get all feature maps
            _ = self.backbone(x)

            # Apply attention to mid-level features
            self.feature_maps[3] = self.cbam_mid(self.feature_maps[3])

            # # Apply SE blocks to higher level features
            # for i in range(3):
            #     self.feature_maps[i + 4] = self.se_blocks[i](self.feature_maps[i + 4])

            # Apply CBAM and Coordinate Attention to high-level features
            high_level_features = self.feature_maps[6]
            high_level_features = self.cbam_high(high_level_features)
            high_level_features = self.coord_attention(high_level_features)

            # Multi-scale feature extraction
            multi_scale_features = self.multi_scale([
                self.feature_maps[3], self.feature_maps[5], high_level_features
            ])

            # Extract region-specific features
            region_features = []
            for i in range(len(self.region_extractors)):
                region_features.append(self.region_extractors[i](high_level_features))

            # Stack features for transformer
            stacked_features = torch.stack(region_features, dim=1)  # (batch_size, num_regions, feature_dim)

            # Region classification
            region_probs = torch.softmax(self.region_classifier(high_level_features), dim=1)

            # Weight region features with classification probabilities
            batch_size = x.size(0)
            num_regions = len(self.region_extractors)
            weighted_features = stacked_features * region_probs.view(batch_size, num_regions, 1)

            # Aggregate features using transformer
            final_features = self.feature_aggregator(weighted_features)

            # Final output
            output = self.final(final_features)

            return output


class GradientCheckpointingWrapper(nn.Module):
    def __init__(self, model):
        super(GradientCheckpointingWrapper, self).__init__()
        self.model = model
        self.use_checkpointing = True

    def forward(self, x):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            return self.model(x)


if __name__ == '__main__':
    # Create model
    model = EnhancedMultiRegionModel(feature_dim=512, num_regions=3)

    # Wrap with gradient checkpointing for memory efficiency during training
    model = GradientCheckpointingWrapper(model)

    # Set to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print("Model output shape:", output.shape)

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {params:,}")