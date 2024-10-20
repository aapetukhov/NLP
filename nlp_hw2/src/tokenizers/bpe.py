import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, DefaultDict, Iterable, Iterator
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import heapq

from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import toolz
import json


DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'


class PretrainedBPE:
    def __init__(self, vocab_size=30000, min_frequency=2):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
        
    def train(self, corpus):
        self.tokenizer.train_from_iterator(corpus, trainer=self.trainer)
        
    def tokenize(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def save(self, path):
        self.tokenizer.save(path)
    
    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)


from collections import defaultdict
from nltk.tokenize import word_tokenize

class BPE():
    #inspired by huggingface guide, but optimized
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = []
        self.token2idx = {}
        self.idx2token = {}

    def fit(self, corpus):
        for text in corpus:
            for word in text:
                self.word_freqs[word] += 1

        alphabet = set()
        for word in self.word_freqs.keys():
            alphabet.update(word)

        vocab = ["</w>"] + sorted(alphabet).copy()
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        while len(vocab) < self.vocab_size:
            pair_freqs = self.calc_pair_freqs()
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.splits = self.merge_pair(*best_pair)
            self.merges.append(best_pair)
            vocab.append(''.join(best_pair))
        
        self.build_token2idx(vocab)
        self.build_idx2token(vocab)

    def build_token2idx(self, vocab):
        self.token2idx = {token: idx for idx, token in enumerate(vocab, start=1)}

    def build_idx2token(self, vocab):
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def calc_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, char1, char2):
        for word in self.word_freqs:
            split = self.splits[word]
            i = 0
            while i < len(split) - 1:
                if split[i] == char1 and split[i + 1] == char2:
                    split = split[:i] + [char1 + char2] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def tokenize(self, text):
        words = word_tokenize(text)
        tokenized_text = []
        self.merges

        for word in words:
            tokens = [c for c in word]
            i = 0
            while i < len(tokens):
                j = 0
                while j < len(tokens) - 1:
                    pair = (tokens[j], tokens[j + 1])
                    if pair in self.merges:
                        tokens[j:j + 2] = [''.join(pair)]
                        if j > 0:
                            j -= 1
                    else:
                        j += 1
                i += 1
            tokenized_text.extend(tokens)
        return tokenized_text

    def tokens_to_indices(self, tokenized_texts):
        assert self.token2idx
        return [self.token2idx[token] for token in tokenized_texts]

    def indices_to_tokens(self, indexed_texts):
        assert self.idx2token
        return [self.idx2token[idx] for idx in indexed_texts]

