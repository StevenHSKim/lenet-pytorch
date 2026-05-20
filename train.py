import torch
import sys

from configs.configs import *
from data.data import get_dataloaders
from lenet_pytorch.lenet_pytorch import LeNet
from lenet_pytorch.loss import MAPLoss
from utils.utils import init_weights

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    model = LeNet().to(device)
    init_weights(model)
    
    loss_function = MAPLoss(j=J)
    
    for n_passes, lr in LR_SCHEDULE:
        for pass_idx in range(n_passes):
            
            # Training
            model.train()
            total_loss = 0.0
            total_passes = sum(p for p, _ in LR_SCHEDULE)
            current_pass = sum(p for p, _ in LR_SCHEDULE[:LR_SCHEDULE.index((n_passes, lr))]) + pass_idx + 1
            
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                
                model.zero_grad()
                output = model(images)
                loss = loss_function(output, targets)
                loss.backward()
                
                
    