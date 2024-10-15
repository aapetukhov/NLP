import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, DefaultDict, Iterable
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import heapq


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
        """
        Initialize BPE with number of merges.
        """
        self.num_merges = num_merges
        self.bpe_merges = []
        self.bpe_pairs = set()
        self.vocab = {}
        self.word_freqs = Counter()
        self.token2idx = {}
        self.idx2token = {}
        self.pair_stats = {}
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

        self.pair_stats = defaultdict(int)
        self.indices = defaultdict(lambda: defaultdict(int))
        for word, freq in self.vocab.items():
            prev_char = word[0]
            for char in word[1:]:
                pair = (prev_char, char)
                self.pair_stats[pair] += freq
                self.indices[pair][word] += 1
                prev_char = char

        self.pairs = [(-freq, pair) for pair, freq in self.pair_stats.items()]
        heapq.heapify(self.pairs) #for optimization

        for i in range(self.num_merges):
            if not self.pairs:
                break

            freq, best_pair = heapq.heappop(self.pairs)
            freq = -freq

            if self.pair_stats.get(best_pair, 0) != freq:
                continue

            self.bpe_merges.append(best_pair)

            changes = self.merge_vocab(best_pair)
            self.update_pair_stats(best_pair, changes)

        self.bpe_pairs = set(''.join(pair) for pair in self.bpe_merges)
        
        self.build_vocab(corpus)

    def build_vocab(self, corpus):
        """
        Build vocabulary from the given corpus.
        """
        token_set: Set[str] = set()
        tokenized_corpus = self.transform(corpus)
        for tokens in tokenized_corpus:
            token_set.update(tokens)
        
        self.token2idx = {token: idx for idx, token in enumerate(sorted(token_set))}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def transform(self, corpus: Iterable[List[str]]) -> List[List[str]]:
        """
        Apply BPE to the given corpus.
        """
        tokenized_texts = []
        for tokens in corpus:
            tokenized_sentence = []
            for token in tokens:
                bpe_tokens = self.apply_bpe(token)
                tokenized_sentence.extend(bpe_tokens)
            tokenized_texts.append(tokenized_sentence)
        return tokenized_texts
    
    def apply_bpe(self, token: str) -> Tuple[str, ...]:
        """
        Apply BPE to one word (token)
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

    def merge_vocab(self, pair_to_merge):
        """
        Merge given pair in the vocab.
        """
        bigram = pair_to_merge
        bigram_str = ''.join(bigram)
        changes = []

        for word in list(self.vocab.keys()):
            if bigram not in zip(word, word[1:]):
                continue

            freq = self.vocab[word]
            chars = list(word)
            i = 0
            new_word = []
            while i < len(chars):
                if i < len(chars) -1 and chars[i] == bigram[0] and chars[i+1] == bigram[1]:
                    new_word.append(bigram_str)
                    i += 2
                else:
                    new_word.append(chars[i])
                    i += 1
            new_word = tuple(new_word)
            self.vocab[new_word] = self.vocab.pop(word)
            changes.append((word, new_word, freq))

        return changes

    def update_pair_stats(self, merged_pair, changes):
        """
        Update pair stats after merge.
        """
        self.pair_stats.pop(merged_pair, None)
        self.indices.pop(merged_pair, None)

        for old_word, new_word, freq in changes:
            prev_char = old_word[0]
            for char in old_word[1:]:
                pair = (prev_char, char)
                self.pair_stats[pair] -= freq
                if self.pair_stats[pair] == 0:
                    self.pair_stats.pop(pair)
                    self.indices.pop(pair, None)
                else:
                    self.indices[pair][old_word] -= 1
                    if self.indices[pair][old_word] == 0:
                        self.indices[pair].pop(old_word)
                prev_char = char

            prev_char = new_word[0]
            for char in new_word[1:]:
                pair = (prev_char, char)
                self.pair_stats[pair] += freq
                self.indices[pair][new_word] = self.indices[pair].get(new_word, 0) + 1
                prev_char = char

            for pair in set(zip(new_word, new_word[1:])):
                freq = self.pair_stats[pair]
                heapq.heappush(self.pairs, (-freq, pair))
