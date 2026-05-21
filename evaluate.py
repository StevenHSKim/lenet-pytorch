import torch
import argparse

from configs.configs import DATA_DIR, BATCH_SIZE, NUM_WORKERS
from data.data import get_dataloaders
from lenet_pytorch.lenet_pytorch import LeNet


def evaluate(weights_path: str = "lenet5.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)

    model = LeNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)

            rbf_output = model(images)
            predictions = rbf_output.argmin(dim=1)  # minimum penalty

            correct += (predictions == targets).sum().item()
            total   += targets.size(0)

    accuracy   = correct / total * 100
    error_rate = 100 - accuracy
    print(f"Test Accuracy : {accuracy:.2f}%")
    print(f"Test Error    : {error_rate:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="results/lenet5.pth")
    args = parser.parse_args()
 
    evaluate(args.weights)