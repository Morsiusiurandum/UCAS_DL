import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from shared import BasicManager


class Manager(BasicManager):
    def trainer(self):
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        print(f'Accuracy: {correct / len(self.test_loader.dataset):.4f}')

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
            img = images[idx].squeeze()  # [1, 28, 28] -> [28, 28]
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Pred: {preds[idx]}, True: {labels[idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    mnist = Manager("config.yaml", model, optimizer, criterion)
    mnist.start_training(is_continue=False, save_path="checkpoint/mnist_cnn.pth")
    mnist.predictions()


if __name__ == "__main__":
    main()
