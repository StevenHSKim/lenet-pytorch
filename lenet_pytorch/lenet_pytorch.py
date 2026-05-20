import torch
from torch import nn
import torch.nn.functional as F

def scaled_tanh(x):
    """f(a) = 1.7159 * tanh (2/3 * a)"""
    return 1.7159 * torch.tanh(2/3 * x)


class SubsamplingLayer(nn.Module):
    """Unlike standard average pooling, applies trainable coefficient and bias after pooling"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 receptive fields are nonoverlapping
        # Trainable per-channel coefficient and bias
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x = self.pool(x) * 4  # AvgPool -> SumPool 
        x = self.weight * x + self.bias
        x = scaled_tanh(x)
        return x
    

class C3PartialConv(nn.Module):
    """Connection between S2 and C3 (Table I)"""
    
    CONNECTION_TABLE = [
        [0,1,2],  # column 0
        [1,2,3],  # column 1
        [2,3,4],  # column 2
        [3,4,5],  # column 3
        [4,5,0],  # column 4
        [5,0,1],  # column 5
        [0,1,2,3],  # column 6
        [1,2,3,4],  # column 7
        [2,3,4,5],  # column 8
        [3,4,5,0],  # column 9
        [4,5,0,1],  # column 10
        [5,0,1,2],  # column 11
        [0,1,3,4],  # column 12
        [1,2,4,5],  # column 13
        [0,2,3,5],  # column 14
        [0,1,2,3,4,5]  # column 15
    ]
    
    def __init__(self):
        super().__init__()        
        self.partial_conv_list = nn.ModuleList()
        for col in self.CONNECTION_TABLE:
            partial_conv = nn.Conv2d(len(col), 1, kernel_size=5)
            self.partial_conv_list.append(partial_conv)
        
    def forward(self, x):
        # x: (B, 6, 14, 14)
        out = []
        for conv, col in zip(self.partial_conv_list, self.CONNECTION_TABLE):
            selected_channels = x[:, col, :, :]
            out.append(conv(selected_channels))
        return torch.cat(out, dim=1)


class EuclideanRadialBasisFunction(nn.Module):
    """Output Layer (RBF): The parameter vectors of these units were chosen by hand and kept fixed at least initially"""
    
    def __init__(self, in_features=84, num_classes=10):
        super().__init__()
        # initialize weights
        self.w = nn.Parameter(
            torch.randn(num_classes, in_features),  # (10, 84)
            requires_grad = False  # initially fixed
        )
        
    def forward(self, x):
        x = (x.unsqueeze(1) - self.w).pow(2).sum(dim=2)  # x.unsqueeze(1) -> (B, 1, 84)
        return x
        
    
    
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2
        self.s2 = SubsamplingLayer(channels=6)
        # C3
        self.c3 = C3PartialConv()
        # S4
        self.s4 = SubsamplingLayer(channels=16)
        # C5
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        # F6
        self.f6 = nn.Linear(120, 84)
        # Output
        self.output = EuclideanRadialBasisFunction(84, 10)

    def forward(self, x):
        x = scaled_tanh(self.c1(x))
        x = self.s2(x)
        x = scaled_tanh(self.c3(x))
        x = self.s4(x)
        x = scaled_tanh(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = scaled_tanh(self.f6(x))
        x = self.output(x)
        return x