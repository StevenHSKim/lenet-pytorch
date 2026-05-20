import torch
from torch import nn

def init_weights(model: nn.Module):
    """Appendix A: uniform(-2.4/Fi, 2.4/Fi) where Fi is Fan in (# of Input Connections)"""
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            nn.init.uniform(module.weight, -2.4/fan_in, 2.4/fan_in)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -2.4/fan_in, 2.4/fan_in)