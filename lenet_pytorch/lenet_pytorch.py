import torch
from torch import nn
from torchvision import datasets, transforms

class SubsamplingLayer(nn.Module):
    """
    Unlike standard average pooling, applies trainable coefficient and bias after pooling
    """
    
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        x = self.weight * x + self.bias
        x = self.sigmoid(x)
        return x
    


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            SubsamplingLayer(6),
            
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            SubsamplingLayer(16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return x