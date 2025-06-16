import random
import sys

sys.path.append("..")
sys.path.append("../shared")

import numpy as np
import torch
import torch.nn as nn

from model import TCNPoetryModel
from shared import get_dataloaders


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, dataloader, criterion, optimizer, device):
    model.trainer()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def show_sample_predictions(model, start_text, idx2char, char2idx, max_gen_len=40, device='cpu'):
    model.eval()
    generated = list(start_text)
    input_seq = torch.tensor([[char2idx.get(c, char2idx['<unk>']) for c in generated]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_gen_len):
            output = model(input_seq)
            last_logits = output[0, -1]  # [vocab_size]
            next_id = torch.multinomial(torch.softmax(last_logits, dim=-1), 1).item()
            next_char = idx2char.get(next_id, '<unk>')
            generated.append(next_char)
            input_seq = torch.cat([input_seq, torch.tensor([[next_id]], device=device)], dim=1)

    print("🌸 自动生成诗句：", "".join(generated))


def main():
    # 🧱 超参数
    embed_size = 256
    num_channels = 512
    num_layers = 4
    kernel_size = 3
    dropout = 0.2
    lr = 0.003
    batch_size = 64
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    train_loader, val_loader, idx2char, char2idx = get_dataloaders('TangPoetry')
    # 🧠 模型构建
    model = TCNPoetryModel(vocab_size=len(char2idx),
                           embed_size=embed_size,
                           num_channels=num_channels,
                           num_layers=num_layers,
                           kernel_size=kernel_size,
                           dropout=dropout).to(device)

    # 🎯 损失函数 + 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 📈 训练主循环
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if epoch % 2 == 0:
            show_sample_predictions(model, start_text="湖光秋月两相和", idx2char=idx2char, char2idx=char2idx, device=device)

    # 💾 模型保存
    torch.save(model.state_dict(), "tcn_poetry.pth")
    print("✅ 模型已保存为 tcn_poetry.pth")


if __name__ == "__main__":
    main()
