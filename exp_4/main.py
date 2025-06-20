import torch
import torch.nn as nn
import torch.optim as optim

from model import BERT2BERTTranslationModel


def train_one_epoch(model, data_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        input_mask = batch['attention_mask'].to(device)

        target_ids = batch['decoder_input_ids'].to(device)
        target_mask = batch['decoder_attention_mask'].to(device)
        target_labels = batch['decoder_labels'].to(device)

        optimizer.zero_grad()
        output_logits = model(input_ids, segment_ids, input_mask, target_ids, target_mask)
        loss = loss_function(output_logits.view(-1, output_logits.size(-1)), target_labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            input_mask = batch['attention_mask'].to(device)

            target_ids = batch['decoder_input_ids'].to(device)
            target_mask = batch['decoder_attention_mask'].to(device)
            target_labels = batch['decoder_labels'].to(device)

            output_logits = model(input_ids, segment_ids, input_mask, target_ids, target_mask)
            loss = loss_function(output_logits.view(-1, output_logits.size(-1)), target_labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    # 模型超参数
    source_vocabulary_size = 300000
    target_vocabulary_size = 300000
    model_dimension = 512
    maximum_sequence_length = 512
    number_of_layers = 6
    number_of_attention_heads = 8
    hidden_dimension = 2048
    dropout_rate = 0.1
    pad_token_id = 0  # 需要根据你的词表设置

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from data_loader import get_dataloader

    train_loader, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word = get_dataloader(
            src_path="dataset/TM-training-set/chinese.txt",
            tgt_path="dataset/TM-training-set/english.txt",
            src_vocab_path="dataset/vocab.zh",
            tgt_vocab_path="dataset/vocab.en"
    )

    # 模型
    model = BERT2BERTTranslationModel(
            src_word2idx.__len__(),
            tgt_word2idx.__len__(),
            model_dimension,
            maximum_sequence_length,
            number_of_layers,
            number_of_attention_heads,
            hidden_dimension,
            dropout_rate
    ).to(device)

    # 损失函数（忽略 pad 位置）
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # 训练过程
    total_epochs = 10
    for epoch in range(total_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        # val_loss = evaluate_model(model, val_loader, loss_function, device)
        # print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")

        torch.save(model.state_dict(), f"bert2bert_epoch{epoch + 1}.pth")

    # 推理
    model.eval()
    # 准备测试输入
    test_sentence = "你好，世界！"
    # 按词分割，如果是按字分词请改为list(test_sentence)
    test_tokens = test_sentence.split()
    if len(test_tokens) == 1:  # 可能是中文未分词，按字分
        test_tokens = list(test_sentence)

    # 将测试句子转换为索引
    test_index = [src_word2idx.get(word, src_word2idx.get('<unk>', 0)) for word in test_tokens]

    input_tensor = torch.tensor(test_index, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    # 假设模型有generate或类似方法，否则用简单的贪心解码
    with torch.no_grad():
        # 这里只做简单贪心解码，具体方法需根据你的模型实现调整
        max_len = 30
        tgt_input = torch.tensor([tgt_word2idx.get('<bos>', 1)], dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_len):
            # 这里假设模型forward支持如下参数，具体请根据你的模型API调整
            output_logits = model(input_tensor, None, None, tgt_input, None)
            next_token_logits = output_logits[0, -1]
            next_token_id = next_token_logits.argmax().item()
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token_id]], device=device)], dim=1)
            if next_token_id == tgt_word2idx.get('<eos>', 2):
                break
        output_ids = tgt_input[0].tolist()
        # 去掉<BOS>和<EOS>

        output_words = [tgt_idx2word.get(idx, '<unk>') for idx in output_ids]
        print(f"Test sentence: {test_sentence}")
        print(f"Tokenized: {test_tokens}")
        print(f"Token IDs: {test_index}")
        print(f"Model output IDs: {output_ids}")
        print(f"Model output words: {' '.join(output_words)}")


if __name__ == "__main__":
    main()

    # from data_loader import get_dataloader
    #
    # dataloader, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word = get_dataloader(
    #         src_path="dataset/TM-training-set/chinese.txt",
    #         tgt_path="dataset/TM-training-set/english.txt",
    #         src_vocab_path="dataset/vocab.zh",
    #         tgt_vocab_path="dataset/vocab.en"
    # )
    #
    # print(f"DataLoader created with {len(dataloader.dataset)} samples.")
    # out = dataloader.dataset[0]
    # print(f"First sample: {dataloader.dataset[0]}")
    # print(f"First sample: {dataloader.dataset[0][0].tolist()} -> \n {dataloader.dataset[0][1].tolist()}")
    # # 打印转码后的内容
    # src_words = [src_idx2word[idx] for idx in dataloader.dataset[0][0].tolist()]
    # tgt_words = [tgt_idx2word[idx] for idx in dataloader.dataset[0][1].tolist()]
    # print(f"Decoded: {' '.join(src_words)} -> {' '.join(tgt_words)}")
    # # 打印每个token对应的单字
    # src_chars = [(src_idx2word[idx], idx) for idx in dataloader.dataset[0][0].tolist()]
    # print(f"Src chars: {src_chars}")
    # tgt_chars = [(tgt_idx2word[idx], idx) for idx in dataloader.dataset[0][1].tolist()]
    # print(f"Tgt chars: {tgt_chars}")
