import os
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from model import BERT2BERTTranslationModel


def trainer(model, data_loader, optimizer, loss_function, device):
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


def evaluate(model, data_loader, loss_function, device):
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


def sample(model, src_word2idx, tgt_idx2word, device):
    ###

    ###
    # 推理
    model.eval()
    # 准备测试输入
    test_sentence = "七 天 时间 里 , 他们 为 赶 时间 , 白天 啃 干馕喝 白开水 , 晚上 居住 在 牧民 的 毡房 里 ."
    test_tokens = test_sentence.split()

    if len(test_tokens) == 1:  # 可能是中文未分词，按字分
        test_tokens = list(test_sentence)

    # 将测试句子转换为索引
    test_index = [src_word2idx.get(word, src_word2idx.get('<UNK>', 0)) for word in test_tokens]
    input_tensor = torch.tensor(test_index, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    with ((torch.no_grad())):
        # 简单贪心解码
        max_len = 30
        tgt_input = torch.tensor([5], dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_len):

            input_token_ids = torch.tensor(test_index, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
            segment_token_type_ids = torch.zeros(len(input_tensor), dtype=torch.long).to(device)

            #  得到模型输出
            output_logits = model(input_token_ids, segment_token_type_ids, None, tgt_input, None)
            next_token_logits = output_logits[0, -1]
            next_token_id = next_token_logits.argmax().item()
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token_id]], device=device)], dim=1)

            # 如果遇到 <EOS>，则停止生成
            if next_token_id == 6:
                break
        output_ids = tgt_input[0].tolist()

        # 将输出索引转换为单词
        output_words = [tgt_idx2word.get(idx, '<UNK>') for idx in output_ids]
        print(f"Test sentence: {test_sentence}")
        print(f"Tokenized: {test_tokens}")
        print(f"Token IDs: {test_index}")
        print(f"Model output IDs: {output_ids}")
        print(f"Model output words: {' '.join(output_words)}")


def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main():
    # 获取 torchrun 设置的 rank/世界大小信息
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)
    # 模型超参数
    model_dimension = 512
    maximum_sequence_length = 512
    number_of_layers = 6
    number_of_attention_heads = 8
    hidden_dimension = 2048
    dropout_rate = 0.2
    pad_token_id = 0  # 需要根据你的词表设置

    # 设备
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from data_loader import get_dataloader

    train_loader, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word = get_dataloader(
            src_path="dataset/TM-training-set/chinese.txt",
            tgt_path="dataset/TM-training-set/english.txt",
            src_vocab_path="dataset/vocab.zh",
            tgt_vocab_path="dataset/vocab.en",
            distributed=True,
            rank=rank,
            world_size=world_size
    )

    # 模型
    model = BERT2BERTTranslationModel(
            len(src_word2idx),
            len(tgt_word2idx),
            model_dimension,
            maximum_sequence_length,
            number_of_layers,
            number_of_attention_heads,
            hidden_dimension,
            dropout_rate
    ).to(device)

    model = DDP(model, device_ids=[rank])
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    total_epochs = 100
    for epoch in range(total_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = trainer(model, train_loader, optimizer, loss_function, device)

        if rank == 0:  # 只在主进程保存
            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), f"bert2bert_epoch{epoch + 1}.pth")
                print(f"Saving model at epoch {epoch + 1}...")
                sample(model.module, src_word2idx, tgt_idx2word, device)

    cleanup()
    # model.load_state_dict(torch.load("bert2bert_epoch10.pth"))


if __name__ == "__main__":
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
    # print(torch.distributed.is_nccl_available())

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
