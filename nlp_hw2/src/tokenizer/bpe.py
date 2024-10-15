import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Any

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


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
    def __init__(self, num_merges : int = 10000):
        """
        Initialize BPE model.

        Parameters
        ----------
        num_merges : int
            Number of merge operations to perform.
        """
        self.num_merges = num_merges
        self.bpe_merges = []
        self.bpe_pairs = set()
        self.vocab = {}
        self.word_freqs = Counter()

    def fit(self, corpus: List[List[str]]):
        for tokens in corpus:
            for word in tokens:
                self.word_freqs[word] += 1

        # бьём слова на символы
        self.vocab = {}
        for word, freq in self.word_freqs.items():
            chars = tuple(word) + ('</w>',)
            self.vocab[chars] = freq

        for _ in range(self.num_merges):
            pairs = self.get_stats()
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.bpe_merges.append(best_pair)

            self.vocab = self.merge_vocab(best_pair)

        self.bpe_pairs = set(''.join(pair) for pair in self.bpe_merges)

    def get_stats(self):
        """
        Count the frequency of all symbol pairs in the vocabulary.

        :return: A dictionary of all symbol pairs and their frequencies.
        """
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def merge_vocab(self, pair_to_merge):
        """
        Merge a pair of symbols in the vocabulary.

        :param pair_to_merge: The pair of symbols to merge.
        :return: The new vocabulary with the merged pair.
        """
        
        vocab_new = {}
        bigram = pair_to_merge
        bigram_str = ''.join(bigram)
        for word, freq in self.vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) -1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(bigram_str)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            vocab_new[tuple(new_word)] = freq
        return vocab_new

    def tokenize_one(self, token):
        """
        Applies BPE to the given token.

        :param token: The token to which apply BPE.
        :return: The token with BPE applied.
        """
        word = tuple(token) + ('</w>',)
        idx = 0
        while idx < len(word)-1:
            pair = (word[idx], word[idx+1])
            bigram = ''.join(pair)
            if bigram in self.bpe_pairs:
                word = word[:idx] + (bigram,) + word[idx+2:]
                idx = max(idx - 1, 0)
            else:
                idx += 1
        if word[-1] == '</w>':
            word = word[:-1]
        return word

    def transform(self, corpus):
        """
        Applies BPE to each token in the given corpus.

        :param corpus: Iterable of iterable of strings (sentences of tokens).
        :return: List of lists of strings (sentences of BPE tokens).
        """
        tokenized_texts = []
        for tokens in corpus:
            tokenized_sentence = []
            for token in tokens:
                bpe_tokens = self.tokenize_one(token)
                tokenized_sentence.extend(bpe_tokens)
            tokenized_texts.append(tokenized_sentence)
        return tokenized_texts

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)

    def save_merges(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in self.bpe_merges:
                f.write(' '.join(pair) + '\n')

    def load_merges(self, filepath):
        self.bpe_merges = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                pair = tuple(line.strip().split())
                self.bpe_merges.append(pair)
        self.bpe_pairs = set(''.join(pair) for pair in self.bpe_merges)
