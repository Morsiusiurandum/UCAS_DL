import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from shared import load_config, get_dataloaders, save_checkpoint, load_checkpoint


def train(model, device, loader, optimizer, criterion):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()


def evaluate(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    print(f'Accuracy: {correct / len(loader.dataset):.4f}')


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
        img = images[idx].squeeze()  # [1, 28, 28] -> [28, 28]
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {preds[idx]}, True: {labels[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    train_loader, test_loader = get_dataloaders(config["dataset"], config["batch_size"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # 加载检查点
    start_epoch = load_checkpoint(model, optimizer, path="checkpoint/mnist_cnn.pth")

    for epoch in range(start_epoch, config["epochs"] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")
        train(model, device, train_loader, optimizer, criterion)
        evaluate(model, device, test_loader)
        save_checkpoint(model, optimizer, epoch, path="checkpoint/mnist_cnn.pth")

    show_sample_predictions(model, device, test_loader, num_images=5)


if __name__ == "__main__":
    main()
