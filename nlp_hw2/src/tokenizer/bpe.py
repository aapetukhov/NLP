import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, DefaultDict, Iterable
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from heapq import heappop as hpop
from heapq import heappush as hpush
from heapq import heapify as heap


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


class BPE:
    def __init__(self, num_merges=10000):
        self.num_merges = num_merges
        self.bpe_merges = []
        self.bpe_pairs = set()

        self.vocab = {}
        self.word_freqs = {}

        self.token2idx = {}
        self.ind2token = {}

        self.stats = {}
        self.pairs = []
        self.indices = {}

    def fit(self, corpus):
        for tokens in corpus:
            for word in tokens:
                self.word_freqs[word] += 1

        self.vocab = {}
        for word, freq in self.word_freqs.items():
            chars = tuple(word) + ('</w>',)
            self.vocab[chars] = freq

        self.stats = defaultdict(int)
        self.indices = defaultdict(lambda: defaultdict(int))

        for word, freq in self.vocab.items():
            prev_char = word[0]

            for char in word[1:]:
                pair = (prev_char, char)
                self.stats[pair] += freq
                self.indices[pair][word] += 1
                prev_char = char

        self.pairs = [(-freq, pair) for pair, freq in self.stats.items()] # -freq тк куча минимальная по дефолту
        heap(self.pairs)

        for _ in range(self.num_merges):
            if not self.pairs:
                break

            freq, best_pair = hpop(self.pairs)
            freq = -freq

            if self.stats.get(best_pair, 0) != freq:
                continue

            self.bpe_merges.append(best_pair)

            seqs = self.merge(best_pair)
            self.update_pairs(best_pair, seqs)

        self.bpe_pairs = set(''.join(pair) for pair in self.bpe_merges)
        
        self.build_vocab(corpus)

    def build_vocab(self, corpus):
        token_set = set()
        tokenized_corpus = self.transform(corpus)
    
        for tokens in tokenized_corpus:
            token_set.update(tokens)
        
        self.token2idx = {token: idx for idx, token in enumerate(sorted(token_set))}
        self.ind2token = {idx: token for token, idx in self.token2idx.items()}

    def transform(self, corpus):
        tokenized_texts = []

        for tokens in corpus:
            tokenized_sentence = []
            for token in tokens:
                bpe_tokens = self.bpe_transform(token)
                tokenized_sentence.extend(bpe_tokens)

            tokenized_texts.append(tokenized_sentence)

        return tokenized_texts
    
    def bpe_transform(self, token: str):
        word = tuple(token) + ('</w>',)
        idx = 0

        while idx < len(word)-1:
            pair = (word[idx], word[idx+1])
            twochar = ''.join(pair)

            if twochar in self.bpe_pairs:
                word = word[:idx] + (twochar,) + word[idx+2:]
                idx = max(idx - 1, 0)
            else:
                idx += 1

        if word[-1] == '</w>':
            word = word[:-1]

        return word

    def merge(self, pair_to_merge):
        twochar = pair_to_merge
        twochar_str = ''.join(twochar)
        seqs = []

        for word in list(self.vocab.keys()):
            if twochar not in zip(word, word[1:]):
                continue

            freq = self.vocab[word]
            chars = list(word)
            i = 0
            new_seq = []
            while i < len(chars):
                if (i < len(chars) -1) and (chars[i] == twochar[0]) and (chars[i+1] == twochar[1]):
                    new_seq.append(twochar_str)
                    i += 2

                else:
                    new_seq.append(chars[i])
                    i += 1
            new_seq = tuple(new_seq)
            self.vocab[new_seq] = self.vocab.pop(word)

            seqs.append((word, new_seq, freq))

        return seqs

    def update_pairs(self, merged_pair, seqs):
        self.stats.pop(merged_pair, None)
        self.indices.pop(merged_pair, None)

        for prev_seq, new_seq, freq in seqs:
            prev_char = prev_seq[0]

            for char in prev_seq[1:]:
                pair = (prev_char, char)
                self.stats[pair] -= freq

                if self.stats[pair] == 0:
                    self.stats.pop(pair)
                    self.indices.pop(pair, None)

                else:
                    self.indices[pair][prev_seq] -= 1
                    if self.indices[pair][prev_seq] == 0:
                        self.indices[pair].pop(prev_seq)

                prev_char = char

            prev_char = new_seq[0]

            for char in new_seq[1:]:
                pair = (prev_char, char)

                self.stats[pair] += freq
                self.indices[pair][new_seq] = self.indices[pair].get(new_seq, 0) + 1

                prev_char = char

            for pair in set(zip(new_seq, new_seq[1:])):
                freq = self.stats[pair]
                hpush(self.pairs, (-freq, pair))
