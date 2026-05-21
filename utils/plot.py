import matplotlib.pyplot as plt


def plot_loss(train_losses: list, val_losses: list, save_path: str = None):
    """Loss vs Epochs"""
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_metric(train_metrics: list, val_metrics: list, metric_name: str = "Accuracy", save_path: str = None):
    """Metric vs Epochs"""
    
    epochs = range(1, len(train_metrics) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_metrics, label=f"Train {metric_name}")
    plt.plot(epochs, val_metrics,   label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()