



import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import seed_everything




class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=13971, two_layers=False, feat_dim=2048, seed=777):
        super(LinearClassifier, self).__init__()

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if two_layers:
          self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_classes)
            )
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)





