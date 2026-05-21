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
                
                
def estimate_hessian(model: nn.Module, loss_fn, data_loader, device, sample_size: int = 500):
    """Appendix C: Gauss-Newton Diagonal Hessian Estimation"""
    
    model.eval()
    hessian_diag = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
 
    count = 0
    for images, targets in data_loader:
        if count >= sample_size:
            break
 
        images, targets = images.to(device), targets.to(device)
        model.zero_grad()
 
        output = model(images)
        loss   = loss_fn(output, targets)
        loss.backward()
 
        # Gauss-Newton
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                hessian_diag[name] += p.grad.data ** 2
 
        count += 1
 
    for name in hessian_diag:
        hessian_diag[name] /= count
 
    return hessian_diag
 
 
def compute_step_sizes(hessian_diag: dict, mu: float):
    """Equation (21)"""
    
    return {name: 1.0 / (mu + h) for name, h in hessian_diag.items()}


def log_model_summary(model: nn.Module, logger):
    logger.info("=" * 50)
    logger.info("Model Summary")
    logger.info("=" * 50)
    
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        logger.info(f"{name:<40}  {str(list(param.shape)):<25}  {n:>8,}")
    
    logger.info("=" * 50)
    logger.info(f"{'Total Parameters':<40}  {'':<25}  {total:>8,}")
    logger.info("=" * 50)