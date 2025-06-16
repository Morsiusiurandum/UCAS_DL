import torch.nn as nn

import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # 只在左边 pad
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCNPoetryModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_channels=512, num_layers=4, kernel_size=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = embed_size if i == 0 else num_channels
            layers.append(ResidualBlock(in_ch, num_channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, vocab_size)

    def forward(self, x):
        # x: [B, T]
        x = self.embed(x).transpose(1, 2)  # [B, E, T]
        y = self.tcn(x)  # [B, C, T]
        y = self.fc(y.transpose(1, 2))  # [B, T, V]
        return y
