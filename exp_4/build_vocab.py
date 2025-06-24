from collections import Counter

# 语料文件路径
corpus_files = ["dataset/TM-training-set/english.txt"]

counter = Counter()
for file in corpus_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

# 可选：添加特殊符号
# PAD: 0,指示填充符号，
# UNK: 1,未知词符号，
# CLS: 2,分类符号，
# SEP: 3,分隔符号，
# MASK: 4,掩码符号
# BOS: 5,开始符号
# EOS：6,结束符号
special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<BOS>", "<EOS>"]

# 词频排序，特殊符号在前
vocab = special_tokens + [token for token, _ in counter.most_common() if token not in special_tokens]

with open("dataset/vocab.en", "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")
