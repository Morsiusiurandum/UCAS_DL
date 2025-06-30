import sys

sys.path.append("..")
sys.path.append("../shared")

import os

os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

from shared import save_checkpoint, load_checkpoint
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from model import BERT2BERTTranslationModel
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction


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

        batch_size, tgt_len = target_ids.size()
        optimizer.zero_grad()

        # constructing a causal mask: prevents the decoder from seeing the future.
        causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()  # [tgt_len, tgt_len]
        causal_mask = ~causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        padding_mask = target_mask.unsqueeze(1).expand(-1, tgt_len, -1)
        decoder_attention_mask = (causal_mask & padding_mask).int()  # 1 indicates the visible location
        output_logits = model(input_ids, segment_ids, input_mask, target_ids, decoder_attention_mask)

        loss = loss_function(output_logits.view(-1, output_logits.size(-1)), target_labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, test_file_path, src_word2idx, tgt_idx2word, device):
    def load_test_pairs(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        sources, targets = [], []
        for i in range(0, len(lines), 3):
            src = lines[i].strip()
            ref = lines[i + 2].strip()
            if src and ref:
                sources.append(src)
                targets.append(ref)
        return sources, targets

    def translate_all(sentences):
        return [generation(model, src_word2idx, tgt_idx2word, sentence, max_len=500, device=device) for sentence in sentences]

    def compute_bleu(references, predictions):
        smoothing_function = SmoothingFunction().method1
        total_bleu = 0
        count = 0
    
        for ref, pred in zip(references, predictions):
            ref_tokens = ref.strip().split()
            pred_tokens = pred.strip().split()
            bleu_score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0, 0, 0, 1), # Only use BLEU-4
                    smoothing_function=smoothing_function
            )
            total_bleu += bleu_score
            count += 1
    
        return total_bleu / count if count > 0 else 0.0
    
    sources, references = load_test_pairs(test_file_path)
    predictions = translate_all(sources)
    bleu = compute_bleu(references, predictions)
    print(f"Test BLEU: {bleu * 100:.2f}")
    return bleu


def generation(model, src_word2idx, tgt_idx2word, test_sentence: str, max_len: int, device):
    model.eval()

    test_tokens = test_sentence.split()
    # May segment, by character
    if len(test_tokens) == 1:
        test_tokens = list(test_sentence)

    test_token_ids = [src_word2idx.get(word, src_word2idx.get('<UNK>')) for word in test_tokens]
    # To Tensor [1, seq_len]
    test_token_ids = torch.tensor(test_token_ids, dtype=torch.long).unsqueeze(0).to(device)

    # The first token in the target sequence, <BOS>
    output_token_ids = torch.tensor([5], dtype=torch.long).unsqueeze(0).to(device)
    with ((torch.no_grad())):

        for _ in range(max_len):

            segment_type = torch.zeros(len(test_token_ids), dtype=torch.long).to(device)
            # Construct causal mask for decoder

            mask_len = len(output_token_ids)
            causal_mask = torch.triu(torch.ones((mask_len, mask_len), device=device), diagonal=1).bool()  # [tgt_len, tgt_len]
            causal_mask = ~causal_mask.unsqueeze(0).expand(1, -1, -1)
            causal_mask = causal_mask.int()

            output_logits = model(test_token_ids, segment_type, None, output_token_ids, causal_mask)
            next_token_logits = output_logits[0, -1]
            next_token_id = next_token_logits.argmax().item()

            # Append the next token to the output sequence
            output_token_ids = torch.cat([output_token_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

            # Stop the build if <EOS> token is generated
            if next_token_id == 6:  # <EOS>
                break
        output_token_ids = output_token_ids[0].tolist()
        # Remove <BOS> and <EOS> tokens
        output_token_ids = output_token_ids[1:]  # remove <BOS>
        output_token_ids = output_token_ids[:-1]  # remove <EOS>
        # Convert the output index to words and output
        output_tokens = [tgt_idx2word.get(token_id, src_word2idx.get('<UNK>')) for token_id in output_token_ids]
        output_sentence = ' '.join(output_tokens)
        return output_sentence


def sample(model, src_word2idx, tgt_word2idx, tgt_idx2word, device) -> str:
    model.eval()

    # 输入
    sentence = "在此 背景 下 , 中国 要 融入 世界 经济 , 有必要 加快 发展 高技术 产业 , 增强 综合 国力 , 改善 这 一 产业 的 国际 分工 地位 ."
    # 候选文本和参考文本
    candidate = "against this background , if china is to be incorporated into the world economy , it is essential that we accelerate the development of high - tech industry , enhance the overall national strength , and improve this industry 's standing in terms of the international division of labor ."
    reference = generation(model, src_word2idx, tgt_idx2word, sentence, max_len=500, device=device)

    # BLEU-4
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()
    smoothing_function = SmoothingFunction().method1  # 平滑函数
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    print("BLEU-4:", bleu_4)
    return reference


def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main():
    # Get the rank/world size information set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)

    model_dimension = 512
    maximum_sequence_length = 512
    number_of_layers = 6
    number_of_attention_heads = 8
    hidden_dimension = 2048
    dropout_rate = 0.2
    pad_token_id = 0

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
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 加载预训练模型

    start_epoch = load_checkpoint(model, optimizer, path="checkpoint/bert2bert_zh2en.pth")

    total_epochs = 500
    if rank == 0:
        print(f"Device set to {device}, rank {rank}, world size {world_size}")
        example = sample(model.module, src_word2idx, tgt_word2idx, tgt_idx2word, device)
        print(f"Example output before training: {example}")
        print(f"Starting training for {total_epochs} epochs,currently on epoch {start_epoch}...")

    for epoch in range(start_epoch, total_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_loss = trainer(model, train_loader, optimizer, loss_function, device)
        # Save only in the main process
        if rank == 0:
            example = sample(model.module, src_word2idx, tgt_word2idx, tgt_idx2word, device)
            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Example: {example}")
            if (epoch + 1) % 50 == 0:
                save_checkpoint(model, optimizer, epoch, "checkpoint/bert2bert_zh2en.pth")
                print(f"Saving model at epoch {epoch + 1}...")
    print("Training complete. Evaluating model...")
    evaluate(model, "dataset/Reference-for-evaluation/Niu.test.reference", src_word2idx, tgt_idx2word, device)
    cleanup()


if __name__ == "__main__":
    main()
