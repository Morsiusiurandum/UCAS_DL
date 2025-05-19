import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import ViTCIFAR10
from shared import load_config, get_dataloaders, save_checkpoint, load_checkpoint


# 训练函数
def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# 测试函数
def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted.eq(labels)).sum().item()
            total += labels.size(0)
    return correct / total


def show_sample_predictions(model, device, test_loader, class_names=None, num_images=5):
    model.eval()
    images, labels, preds = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            images.extend(data.cpu())
            labels.extend(target.cpu())
            preds.extend(pred.cpu())
            if len(images) >= num_images:
                break

    indices = random.sample(range(len(images)), num_images)
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        img = images[idx].permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        title = f"Pred: {preds[idx]}\nTrue: {labels[idx]}"
        if class_names:
            title = f"Pred: {class_names[preds[idx]]}\nTrue: {class_names[labels[idx]]}"
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTCIFAR10().to(device)
    train_loader, test_loader = get_dataloaders(config["dataset"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    start_epoch = load_checkpoint(model, optimizer, path="checkpoint/vit_cifar10_final.pth")
    # 训练和评估模型
    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train(model, device, train_loader, optimizer, criterion)
        acc = evaluate(model, device, test_loader)
        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Test Acc={acc:.4f}")
        save_checkpoint(model, optimizer, epoch, path=f"checkpoint/vit_cifar10_{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, path="checkpoint/vit_cifar10_final.pth")


# 主流程
if __name__ == '__main__':
    main()
