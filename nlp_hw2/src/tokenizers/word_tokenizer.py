from nltk.corpus import stopwords
from collections import defaultdict, Counter
from typing import List
import nltk
import pandas as pd

STOP_WORDS = set(stopwords.words("english"))


class WordTokenizer:
    def __init__(self, min_word_freq: int = 3, max_doc_freq: float = 0.5):
        self.min_word_freq = min_word_freq
        self.max_doc_freq = max_doc_freq
        self.word_counts = None
        self.doc_freq = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None

    def fit(self, tokenized_texts):
        num_docs = len(tokenized_texts)
        max_doc_freq = self.max_doc_freq * num_docs

        self.word_counts = Counter()
        self.doc_freq = Counter()
        for tokens in tokenized_texts:
            self.word_counts.update(tokens)
            self.doc_freq.update(set(tokens))

        self.vocab = {word for word in self.word_counts
                      if self.word_counts[word] >= self.min_word_freq and self.doc_freq[word] <= max_doc_freq}

        self.word2idx = {word: idx for idx, word in enumerate(sorted(self.vocab), start=1)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def tokens_to_indices(self, tokenized_texts):
        return [[self.word2idx[word] for word in tokens if word in self.vocab] for tokens in tokenized_texts]

    def indices_to_tokens(self, indexed_texts):
        return [[self.idx2word[idx] for idx in indices] for indices in indexed_texts]
    