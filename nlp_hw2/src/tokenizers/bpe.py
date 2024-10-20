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


class BPE():
    # inspired by huggingface bpe notes, but wrapped and optimized
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}

    def fit(self, corpus):
        # corpus - list of lists of words, i.e. tokenized texts after word_tokenize
        for text in corpus:
            for word in text:
                self.word_freqs[word] += 1

        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()

        vocab = ["</w>"] + alphabet.copy()

        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        while len(vocab) < self.vocab_size:

            pair_freqs = self.calc_pair_freqs()

            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            self.splits = self.merge_pair(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
        return self.merges

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
            if len(split) == 1:
                continue
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
        splits_text = [[c for c in word] for word in words]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits_text[idx] = split
        result = sum(splits_text, [])
        return result
