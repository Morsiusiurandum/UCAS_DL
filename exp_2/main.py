import sys

sys.path.append("..")
sys.path.append("../shared")

import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import ViTCIFAR10
from shared import BasicManager
from tqdm import tqdm


class Manager(BasicManager):
    def trainer(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in tqdm(self.train_loader, desc="Training"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # ======= 精度统计 =======
            preds = outputs.argmax(dim=1)  # 取最大值对应的类
            correct += (preds.eq(labels)).sum().item()
            total += labels.size(0)

        loss = total_loss / len(self.train_loader)
        acc = correct / total * 100  # 转换为百分比
        return loss, acc

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds.eq(labels)).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.test_loader)
        acc = correct / total * 100
        return avg_loss, acc

    def predictions(self):
        num_images = 5
        self.model.eval()
        images, labels, preds = [], [], []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    model = ViTCIFAR10()
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()
    cifar10 = Manager("config.yaml", model, optimizer, criterion)
    cifar10.start_training(is_continue=True, save_path="checkpoint/vit_cifar10.pth")
    cifar10.predictions()


if __name__ == '__main__':
    main()
