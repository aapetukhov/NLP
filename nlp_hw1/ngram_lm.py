from collections import defaultdict
from typing import List, Tuple



def int_dict():
    return defaultdict(int)


class NGramLanguageModel:
    def __init__(self, corpus: List[List[str]], n: int):
        """
        corpus: корпус текстов, разбитых на токены
        n: для т-грамм
        """
        self.n = n
        self.ngram_freq = defaultdict(int_dict)  # вложенные словари для поиска ща О(1)
        self.prefix_freq = defaultdict(int)

        for seq in corpus:
            padded_seq = ["<PAD>"] * n + seq  # как в лекции
            for i in range(len(padded_seq) - n):
                # ngram = tuple(padded_seq[i: i + n + 1])
                prefix = tuple(padded_seq[i : i + n])
                next_word = padded_seq[i + n]  # последнее слово n-граммы
                self.ngram_freq[prefix][next_word] += 1
                self.prefix_freq[prefix] += 1

    def get_next_words_and_probs(
        self, prefix: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Возвращает список слов, которые могут идти после prefix,
        а также список вероятностей этих слов
        """
        padded_prefix = ["<PAD>"] * (self.n - len(prefix)) + prefix
        prefix_tuple = tuple(padded_prefix[-(self.n) :])

        next_words = []
        probs = []

        if prefix_tuple in self.ngram_freq:
            total_count = self.prefix_freq[prefix_tuple]
            for next_word, count in self.ngram_freq[prefix_tuple].items():
                next_words.append(next_word)
                probs.append(count / total_count)

        return next_words, probs
