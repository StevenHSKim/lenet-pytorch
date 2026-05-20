import torch
from torch import nn

class MAPLoss(nn.Module):
    """Equation (9) on paper"""
    
    def __init__(self, j, float=0.1):
        super().__init__()
        self.j = j
        
    def forward(self, rbf_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = rbf_output.size(0)
        correct_penalty = rbf_output[range(batch_size), targets]
 
        competitive_term = torch.log(torch.exp(torch.tensor(self.j)) + torch.sum(torch.exp(-rbf_output), dim=1))
 
        loss = correct_penalty + competitive_term
        return loss.mean()