import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationModel(nn.Module):
    def __init__(self, base_encoder, hidden_dim=512, num_classes=10):
        super().__init__()
        self.base_encoder = base_encoder
        in_dim = self.base_encoder.fc[3].weight.shape[0]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.base_encoder(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
