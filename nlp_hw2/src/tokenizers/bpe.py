import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, DefaultDict, Iterable, Iterator
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import heapq

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


class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=8192, pct_bpe=0.5, word_tokenizer=None,
                 silent=True, ngram_min=2, ngram_max=2, required_tokens=None,
                 strict=False, lowercase=True,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        self.UNK = UNK
        self.PAD = PAD
        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict
        self.lowercase = lowercase

    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

    def byte_pair_counts(self, words):
        # type: (Encoder, Iterable[str]) -> Iterable[Counter]
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()  # type: Counter
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts

    def count_tokens(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Count tokens into a BPE vocab """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer, sentences)))
        for token in set(self.required_tokens or []):
            word_counts[token] = int(2**63)
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()  # type: Counter
        for token in {self.SOW, self.EOW}:
            vocab[token] = int(2**63)
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, text):
        # type: (Encoder, Iterable[str]) -> None
        """ Learn vocab from text. """
        if self.lowercase:
            _text = [l.lower().strip() for l in text]
        else:
            _text = [l.strip() for l in text]
        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(_text)

        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                           if word not in self.word_vocab]
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)

        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab):
        # type: (int, Dict[str, int]) -> None
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.UNK)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize(self, sentence):
        # type: (Encoder, str) -> List[str]
        """ Split a sentence into word and subword tokens """
        if self.lowercase:
            word_tokens = self.word_tokenizer(sentence.lower().strip())
        else:
            word_tokens = self.word_tokenizer(sentence.strip())
        tokens = []
        for word_token in word_tokens:
            if word_token in self.word_vocab:
                tokens.append(word_token)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def transform(self, sentences, reverse=False, fixed_length=None):
        # type: (Encoder, Iterable[str], bool, int) -> Iterable[List[int]]
        """ Turns space separated tokens into vocab idxs """
        direction = -1 if reverse else 1
        for sentence in self._progress_bar(sentences):
            in_subword = False
            encoded = []
            if self.lowercase:
                tokens = list(self.tokenize(sentence.lower().strip()))
            else:
                tokens = list(self.tokenize(sentence.strip()))
            for token in tokens:
                if in_subword:
                    if token in self.bpe_vocab:
                        if token == self.EOW:
                            in_subword = False
                        encoded.append(self.bpe_vocab[token])
                    else:
                        encoded.append(self.word_vocab[self.UNK])
                else:
                    if token == self.SOW:
                        in_subword = True
                        encoded.append(self.bpe_vocab[token])
                    else:
                        if token in self.word_vocab:
                            encoded.append(self.word_vocab[token])
                        else:
                            encoded.append(self.word_vocab[self.UNK])

            if fixed_length is not None:
                encoded = encoded[:fixed_length]
                while len(encoded) < fixed_length:
                    encoded.append(self.word_vocab[self.PAD])

            yield encoded[::direction]

    def inverse_transform(self, rows):
        # type: (Encoder, Iterable[List[int]]) -> Iterator[str]
        """ Turns token indexes back into space-joined text. """
        for row in rows:
            words = []

            rebuilding_word = False
            current_word = ''
            for idx in row:
                if self.inverse_bpe_vocab.get(idx) == self.SOW:
                    if rebuilding_word and self.strict:
                        raise ValueError('Encountered second SOW token before EOW.')
                    rebuilding_word = True

                elif self.inverse_bpe_vocab.get(idx) == self.EOW:
                    if not rebuilding_word and self.strict:
                        raise ValueError('Encountered EOW without matching SOW.')
                    rebuilding_word = False
                    words.append(current_word)
                    current_word = ''

                elif rebuilding_word and (idx in self.inverse_bpe_vocab):
                    current_word += self.inverse_bpe_vocab[idx]

                elif rebuilding_word and (idx in self.inverse_word_vocab):
                    current_word += self.inverse_word_vocab[idx]

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx])

                elif idx in self.inverse_bpe_vocab:
                    if self.strict:
                        raise ValueError("Found BPE index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_bpe_vocab[idx])

                else:
                    raise ValueError("Got index {} that was not in word or BPE vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')

    def vocabs_to_dict(self, dont_warn=False):
        # type: (Encoder, bool) -> Dict[str, Dict[str, int]]
        """ Turns vocab into dict that is json-serializeable """
        if self.custom_tokenizer and not dont_warn:
            print("WARNING! You've specified a non-default tokenizer.  You'll need to reassign it when you load the "
                  "model!")
        return {
            'byte_pairs': self.bpe_vocab,
            'words': self.word_vocab,
            'kwargs': {
                'vocab_size': self.vocab_size,
                'pct_bpe': self.pct_bpe,
                'silent': self._progress_bar is iter,
                'ngram_min': self.ngram_min,
                'ngram_max': self.ngram_max,
                'required_tokens': self.required_tokens,
                'strict': self.strict,
                'EOW': self.EOW,
                'SOW': self.SOW,
                'UNK': self.UNK,
                'PAD': self.PAD,
            }
        }

    def save(self, outpath, dont_warn=False, encoding=None, ensure_ascii=True, indent=2):
        """ Serializes and saves encoder to provided path """
        with open(outpath, 'w', encoding=encoding) as outfile:
            json.dump(self.vocabs_to_dict(dont_warn), outfile, ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, vocabs):
        # type: (Any, Dict[str, Dict[str, int]]) -> Encoder
        """ Load encoder from dict produced with vocabs_to_dict """
        encoder = Encoder(**vocabs['kwargs'])
        encoder.word_vocab = vocabs['words']
        encoder.bpe_vocab = vocabs['byte_pairs']

        encoder.inverse_bpe_vocab = {v: k for k, v in encoder.bpe_vocab.items()}
        encoder.inverse_word_vocab = {v: k for k, v in encoder.word_vocab.items()}

        return encoder

    @classmethod
    def load(cls, in_path):
        # type: (Any, str) -> Encoder
        """ Loads an encoder from path saved with save """
        with open(in_path) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj)


class BPEEncoder:
    def __init__(
            self,
            vocab_size: int = 8000,
            bpe_fraction: float = 0.4):
        #TODO: add asserts
        self.vocab_size = vocab_size
        self.bpe_vocab_size = int(vocab_size * bpe_fraction)
        self.word_vocab_size = vocab_size - self.bpe_vocab_size

    def fit(self, texts):
        self.word_vocab : dict = self.make_word_vocab(texts)
        self.word_idx2token = {idx: token for token, idx in self.word_vocab.items()}

        leftovers = [
            word for word in toolz.concat(
                map(self.word_tokenizer, texts) #TODO: add tokenizer logic
            )
            if word not in self.word_vocab
        ]
        # из слов, оставшихся в leftovers, делаем bpe
        self.bpe_vocab : dict = self.make_bpe_vocab(leftovers)
        self.bpe_idx2token = {idx: token for token, idx in self.bpe_vocab.items()}

    def make_word_vocab(self, texts):
        words = [word for word in toolz.concat(map(self.word_tokenizer, texts))]
        vocab = Counter(words)
        
        for i, token in enumerate(self.special_tokens):
            vocab[token] = int(2**30 + i)

        sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
        return {
            word: idx for idx, (word, count) in enumerate(sorted_vocab[ : self.word_vocab_size])
        }

    
    def make_bpe_vocab(self, words: Iterable[str]) -> Dict:
        vocab = Counter()
        # vocab[self.SOW] = int(2**30)
        # vocab[self.EOW] = int(2**30 + 1)

        for idx, bpc in enumerate(
            self.bpc(words)
        ):
            for bytepair, count in bpc.items():
                vocab[bytepair] += count

            # TODO: maybe add vocab reduction based on idx?
        
        sorted_counts = sorted(vocab.items(), key=lambda x: -x[1])[ : self.bpe_vocab_size]
        
        return {
            byte_pair: idx + self.word_vocab_size for idx, (byte_pair, count) in enumerate(
                sorted_counts
            )
        }
    
    def bpc(self, words: Iterable[str]) -> Iterable[Counter]:
        for token, count in tqdm(
            self.count_tokens(words).items()
        ):
            bytepair_counts = Counter()
            for bigram in token.split(" "):
                bytepair_counts[bigram] += count
            for ngram_size in range(2, min([2, len(token)]) + 1):
                bigrams = [
                    "".join(bigram) for bigram in toolz.sliding_window(
                        ngram_size, token.split(" ")
                    )
                ]
                for bigram in bigrams:
                    bytepair_counts["".join(bigram)] += count