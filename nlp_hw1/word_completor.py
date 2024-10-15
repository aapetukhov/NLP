from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from prefix_tree import PrefixTree


class WordCompletor:
    def __init__(self, corpus: List[List[str]]):
        """
        corpus: list – корпус текстов, разбитых на токены. на самом деле List[List[str]] - всё обман
        """
        self.freqs = defaultdict(int)  # чтоб автоматом сразу были нули и без говнокода
        self.total_words = 0
        for sentence in corpus:
            for word in sentence:
                self.freqs[word] += 1
                self.total_words += 1

        vocabulary = list(self.freqs.keys())

        self.prefix_tree = PrefixTree(vocabulary)

    def get_words_and_probs(self, prefix: str) -> Tuple[List[str], List[float]]:
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно).
        """
        suitable = self.prefix_tree.search_prefix(prefix)

        words = []
        probs = []
        for word in suitable:
            words.append(word)
            probs.append(
                self.freqs[word] / self.total_words if self.total_words > 0 else 0
            )

        zipped = sorted(zip(words, probs), key=lambda x: x[1], reverse=True)
        words, probs = zip(*zipped)

        return words, probs
