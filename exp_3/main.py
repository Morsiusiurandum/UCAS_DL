import random
import sys

sys.path.append("..")
sys.path.append("../shared")

import numpy as np
import torch
import torch.nn as nn

from model import TCNPoetryModel
from shared import get_dataloaders, show_log, save_checkpoint, load_checkpoint


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
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


def show_sample_predictions(model, start_text, idx2char, char2idx, max_gen_len=125, device=torch.device("cpu")):
    model.eval()
    generated = ['<START>'] + list(start_text)
    char_ids = [char2idx.get(c, char2idx['<unk>']) for c in generated]
    char_ids_2d = [char_ids]
    input_seq = torch.tensor(char_ids_2d, dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_gen_len):
            output = model(input_seq)
            last_logits = output[0, -1]
            next_id = torch.multinomial(torch.softmax(last_logits, dim=-1), 1).item()
            next_char = idx2char.get(next_id, '<unk>')
            generated.append(next_char)
            if next_char == 8290:
                break
            input_seq = torch.cat([input_seq, torch.tensor([[next_id]], device=device)], dim=1)

    return "".join(generated)


def main():
    # 超参数
    embed_size = 256
    num_channels = 512
    num_layers = 6
    kernel_size = 3
    dropout = 0.2
    lr = 0.0005
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    train_loader, val_loader, idx2char, char2idx = get_dataloaders('TangPoetry')

    # 向idx2char和char2idx添加<unk>标记
    length = len(char2idx)
    idx2char[length] = '<unk>'
    char2idx['<unk>'] = len(char2idx)

    # 🧠 模型构建
    model = TCNPoetryModel(vocab_size=len(char2idx),
                           embed_size=embed_size,
                           num_channels=num_channels,
                           num_layers=num_layers,
                           kernel_size=kernel_size,
                           dropout=dropout).to(device)

    # 损失函数 + 优化器
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['</s>'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 模型加载

    start_epoch = load_checkpoint(model, optimizer, path="checkpoint/tcn_poetry.pth")
    example_1 = show_sample_predictions(model, "海上生明月", idx2char, char2idx, device=device)
    example_2 = show_sample_predictions(model, "城阙辅三秦", idx2char, char2idx, device=device)
    example_3 = show_sample_predictions(model, "前不见古人", idx2char, char2idx, device=device)
    example_4 = show_sample_predictions(model, "飞桥隔野烟", idx2char, char2idx, device=device)
    print(f"Examples:\n1: {example_1}\n2: {example_2}\n3: {example_3}\n4: {example_4}")

    # 训练主循环
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        example = show_sample_predictions(model, None, char2idx, max_gen_len=150, device=device)
        show_log(train_loss, val_loss, epoch, example)
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, path="checkpoint/tcn_poetry.pth")


if __name__ == "__main__":
    main()
