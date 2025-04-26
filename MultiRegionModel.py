import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary

class MultiRegionModel(nn.Module):
    def __init__(self, feature_dim = 512):
        super().__init__()
        self.backbone = resnet50(weights = ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.eye_region = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        self.forehead_region = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        self.allface_region = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        self.reg_class = nn.Linear(in_features, 3)
        self.final = nn.Linear(feature_dim*3, feature_dim)


    def forward(self,x):
        feature = self.backbone(x)
        eye_feat = self.eye_region(feature)
        forehead_feat = self.forehead_region(feature)
        allface_feat = self.allface_region(feature)

        # Attention
        region_probs = torch.softmax(self.reg_class(feature), dim=1)
        combined = torch.cat([
            region_probs[:, 0:1] * eye_feat,
            region_probs[:, 1:2] * forehead_feat,
            region_probs[:, 2:3] * allface_feat
        ], dim=1)

        return self.final(combined), region_probs



if __name__ == '__main__':
    model = MultiRegionModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    output, pro = model(dummy_input)
    print("Output shape:", output.shape)
    print(pro)