import torch.nn as nn
import torch.nn.functional as f


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # layer 1: Conv2D + ReLU,input: 28x28x1, output: 26x26x32
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # layer 2: Conv2D + ReLU, input: 26x26x32, output: 24x24x64
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
