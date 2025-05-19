import os

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataloaders(datasets_name, batch_size=64):
    if datasets_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    elif datasets_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT 要求更大尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset name. Use 'MNIST' or 'CIFAR10'.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from: {path}, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at: {path}, starting from scratch.")
        return 1
