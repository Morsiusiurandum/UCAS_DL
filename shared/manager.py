import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from .utils import load_checkpoint, save_checkpoint, get_dataloaders


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BasicManager:

    def __init__(self, config_name: str, model, optimizer, criterion):
        self.config_name = config_name
        self.config = load_config(config_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.test_loader = get_dataloaders(self.config["dataset"], self.config["batch_size"])

        self.optimizer.lr = self.config["learning_rate"]
        self.scheduler = CosineAnnealingLR(optimizer, T_max=self.config["epochs"])

    def trainer(self):
        raise NotImplementedError("This function has not yet implemented.")

    def evaluate(self):
        raise NotImplementedError("This function has not yet implemented.")

    def predictions(self):
        raise NotImplementedError("This function has not yet implemented.")

    def start_training(self, is_continue: bool, save_path: str):

        if is_continue:
            start_epoch = load_checkpoint(self.model, self.optimizer, path=save_path)
        else:
            start_epoch = 0

        epochs = self.config["epochs"]
        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc = self.trainer()
            test_loss, test_acc = self.evaluate()
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
            save_checkpoint(self.model, self.optimizer, epoch, path=save_path)
            self.scheduler.step()
