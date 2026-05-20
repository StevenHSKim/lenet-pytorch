from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    """white: -0.1, black: 1.175"""
    
    return transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIST standard normalization
    ])
    

def get_dataloaders(data_dir, batch_size, num_workers):
    transform = get_transforms()
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, test_dataloader