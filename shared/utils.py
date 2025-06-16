import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, random_split
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
        # 训练集增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先填充4像素再随机裁剪
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 增强色彩抖动的强度
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),  # CIFAR-10 官方均值方差
        ])
        # 测试集只做标准化
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=test_transform)

    elif datasets_name == 'TangPoetry':
        full_dataset = TangPoetryDataset("dataset/tang.npz")
        total_size = len(full_dataset)
        valid_size = int(total_size * 0.1)  # 10% 用于验证集
        train_size = total_size - valid_size
        train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, full_dataset.ix2word, full_dataset.word2ix

    else:
        raise ValueError("Unsupported dataset name.")

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


class TangPoetryDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.data = data['data']  # shape: (57580, 125)
        self.ix2word = data['ix2word'].item()
        self.word2ix = data['word2ix'].item()  # 注意这里要用 .item() 转换为 dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, :-1], dtype=torch.long)  # 输入：前124个字
        y = torch.tensor(self.data[idx, 1:], dtype=torch.long)  # 输出：后124个字（右移1）
        return x, y
