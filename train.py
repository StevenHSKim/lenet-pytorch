import torch
import os
from datetime import datetime

from configs.configs import *
from data.data import get_dataloaders
from lenet_pytorch.lenet_pytorch import LeNet
from lenet_pytorch.loss import MAPLoss
from utils.utils import init_weights, estimate_hessian, compute_step_sizes
from utils.logging import get_logger
from utils.plot import plot_loss

os.makedirs("results", exist_ok=True)
log = get_logger()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    
    train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    model = LeNet().to(device)
    init_weights(model)
    
    loss_function = MAPLoss(j=J)
    
    train_losses  = []
    total_passes  = sum(p for p, _ in LR_SCHEDULE)
    
    for n_passes, lr in LR_SCHEDULE:
        for pass_idx in range(n_passes):
            
            hessian_diag = estimate_hessian(model, loss_function, train_loader, device, HESSIAN_SAMPLE_SIZE)
            step_sizes   = compute_step_sizes(hessian_diag, MU)
            
            # Training
            model.train()
            total_loss = 0.0
            current_pass = sum(p for p, _ in LR_SCHEDULE[:LR_SCHEDULE.index((n_passes, lr))]) + pass_idx + 1
            
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                
                model.zero_grad()
                output = model(images)
                loss = loss_function(output, targets)
                loss.backward()
                
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None and name in step_sizes:
                            param -= lr * step_sizes[name] * param.grad
 
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            log.info(f"[Pass {current_pass}/{total_passes}] lr={lr:.5f}  loss={avg_loss:.4f}")
 
    torch.save(model.state_dict(), f"results/lenet5_{timestamp}.pth")
    log.info(f"Saved: results/lenet5_{timestamp}.pth")
 
    plot_loss(train_losses, save_path=f"results/loss_{timestamp}.png")
    log.info(f"Saved: results/loss_{timestamp}.png")
                
if __name__ == "__main__":
    train()