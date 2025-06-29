import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class TextDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab_path, tgt_vocab_path, pad_token='<PAD>', bos_token='<BOS>', eos_token='<EOS>',
                 unk_token='<UNK>'):
        # 加载词表
        self.src_word2idx, self.src_idx2word = self._load_vocab(src_vocab_path)
        self.tgt_word2idx, self.tgt_idx2word = self._load_vocab(tgt_vocab_path)
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        # 兼容原有代码
        self.src_vocab = self.src_word2idx
        self.tgt_vocab = self.tgt_word2idx
        # 加载语料
        self.src_lines = self._load_lines(src_path)
        self.tgt_lines = self._load_lines(tgt_path)

    def _load_vocab(self, vocab_path):
        word2idx = {}
        idx2word = {}
        with open(vocab_path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                word = line.strip()
                word2idx[word] = idx
                idx2word[idx] = word
        return word2idx, idx2word

    def _load_lines(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        # 源语言
        src_tokens = [self.src_word2idx.get(w, self.src_word2idx.get(self.unk_token)) for w in self.src_lines[idx].split()]
        # 目标语言加BOS/EOS
        tgt_tokens = [self.tgt_word2idx.get(self.bos_token)] + \
                     [self.tgt_word2idx.get(w, self.tgt_word2idx.get(self.unk_token)) for w in self.tgt_lines[idx].split()] + \
                     [self.tgt_word2idx.get(self.eos_token)]
        
        # 处理目标语言的长度
        
        # decoder_input_ids（去掉最后一个token），decoder_labels（去掉第一个token）
        decoder_input_ids = tgt_tokens[:-1]
        decoder_labels = tgt_tokens[1:]
        return {
                'input_ids'             : torch.tensor(src_tokens, dtype=torch.long),
                'segment_ids'           : torch.zeros(len(src_tokens), dtype=torch.long),  # 若无segment信息，填0
                'attention_mask'        : torch.ones(len(src_tokens), dtype=torch.long),
                'decoder_input_ids'     : torch.tensor(decoder_input_ids, dtype=torch.long),
                'decoder_attention_mask': torch.ones(len(decoder_input_ids), dtype=torch.long),
                'decoder_labels'        : torch.tensor(decoder_labels, dtype=torch.long)
        }


def collate_fn(batch):
    # batch是dict的list
    keys = batch[0].keys()
    out = {}
    for key in keys:
        if key in ['input_ids', 'segment_ids', 'attention_mask']:
            pad_value = 0
        else:
            pad_value = 0
        tensors = [item[key] for item in batch]
        out[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
    return out


def get_dataloader(
        src_path,
        tgt_path,
        src_vocab_path,
        tgt_vocab_path,
        batch_size=128,
        shuffle=True,
        distributed=False,
        rank=0,
        world_size=1
):
    dataset = TextDataset(src_path, tgt_path, src_vocab_path, tgt_vocab_path)

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=False
    )

    return dataloader, dataset.src_word2idx, dataset.src_idx2word, dataset.tgt_word2idx, dataset.tgt_idx2word
