import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, DefaultDict, Iterable
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from heapq import heappop
from heapq import heappush
from heapq import heapify


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
        self.bpe_codes = {}

    def fit(self, corpus):        
        word_freqs = Counter()
        for sentence in corpus:
            word_freqs.update(sentence)

        vocab = {tuple(word) + ('</w>',): freq for word, freq in word_freqs.items()}

        for i in range(self.num_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.bpe_codes[best_pair] = len(self.bpe_codes)

            new_vocab = {}
            for word, freq in vocab.items():
                new_word = self._merge_pair(word, best_pair)
                new_vocab[new_word] = freq
            vocab = new_vocab

    def _merge_pair(self, word, pair):
        pair_str = ''.join(pair)
        i = 0
        new_word = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(pair_str)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def transform(self, corpus):
        transformed_corpus = []
        for sentence in corpus:
            transformed_sentence = []
            for word in sentence:
                symbols = list(word) + ['</w>']
                word_symbols = self._encode_word(symbols)
                if word_symbols[-1] == '</w>':
                    word_symbols = word_symbols[:-1]
                transformed_sentence.extend(word_symbols)
            transformed_corpus.append(transformed_sentence)
        return transformed_corpus

    def _encode_word(self, symbols):
        pairs = self._get_pairs(symbols)
        while True:
            min_pair = None
            for pair in pairs:
                if pair in self.bpe_codes:
                    if not min_pair or self.bpe_codes[pair] < self.bpe_codes[min_pair]:
                        min_pair = pair
            if not min_pair:
                break
            symbols = self._merge_pair(symbols, min_pair)
            pairs = self._get_pairs(symbols)
        return symbols

    def _get_pairs(self, symbols):
        return [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
