#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
import loader as loader
from train import *
from loader import *
from model import NERModel
from torch.utils.data import DataLoader

args = argparse.ArgumentParser()

args.add_argument("--dataset", default="wnut16", choices=["GMB", "wnut16"])
args.add_argument("--use_gpu", action="store_true", default=True)
args.add_argument("--word_dim", type=int, default=100)
args.add_argument("--pre_emb", default="../Data/src/glove.6B.100d.txt")
args.add_argument("--lstm_dim", type=int, default=300)
args.add_argument("--epoch", type=int, default=100)
args.add_argument("--use_crf", action="store_true", default=False)
args.add_argument("--batch_size", type=int, default=32)
args.add_argument("--num_layers", type=int, default=2)
args.add_argument("--num_workers", type=int, default=4)
args = args.parse_args([])

args.train = "../Data/ner/" + args.dataset + "/train"
args.test = "../Data/ner/" + args.dataset + "/test"
args.dev = "../Data/ner/" + args.dataset + "/dev"
use_gpu = args.use_gpu


# prepare the dataset

train_sentences = loader.load_sentences(args.train)
dev_sentences = loader.load_sentences(args.dev)
test_sentences = loader.load_sentences(args.test)

word2id, id2word = word_mapping(
    train_sentences, test_sentences, dev_sentences)
tag2id, id2tag = tag_mapping(
    train_sentences, test_sentences, dev_sentences)

train_set = NERDataset(train_sentences, word2id, tag2id)
test_set = NERDataset(test_sentences, word2id, tag2id)
dev_set = NERDataset(dev_sentences, word2id, tag2id)

train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=train_set.collate_fn)

test_data = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=test_set.collate_fn)

dev_data = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=dev_set.collate_fn)


all_word_embeds = {}
word_dim = args.word_dim
if args.pre_emb:
    for i, line in enumerate(open(args.pre_emb, "r", encoding="utf-8")):
        s = line.strip().split()
        word_dim = len(s) - 1
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
    print("Loaded %i pretrained embeddings." % len(all_word_embeds))

word_embeds = np.random.uniform(-np.sqrt(0.06),
                                np.sqrt(0.06), (len(word2id), word_dim))

for w in word2id:
    if w in all_word_embeds:
        word_embeds[word2id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word2id[w]] = all_word_embeds[w.lower()]



model = NERModel(
        vocab_size=len(word2id),
        tag_to_ix=tag2id,
        embedding_dim=word_dim,
        hidden_dim=args.lstm_dim,
        num_laters=args.num_layers,
        pre_word_embeds=word_embeds,
        use_gpu=args.use_gpu,
        use_crf=args.use_crf,
    )


if args.use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)
train(model, args.epoch, train_data, dev_data,
        test_data, use_gpu=args.use_gpu, id_to_tag=id2tag)
