from typing import Any, Dict, List, Optional, Tuple, Union

from ngram_lm import NGramLanguageModel
from prefix_tree import PrefixTree, PrefixTreeNode
from suggester import TextSuggestion, WordPieceTextSuggestion
from tokenizers import BertWordPieceTokenizer
from word_completor import WordCompletor


class SuggestionModel:
    def __init__(
        self,
        corpus: List[List[str]],
        ngram_order: int,
        tokenizer: BertWordPieceTokenizer,
    ):
        self.word_completor = WordCompletor(corpus)
        print("Обучился дополнятель слов")

        self.n_gram_model = NGramLanguageModel(corpus, ngram_order)
        print("Обучилась n-gram модель")

        self.tokenizer = tokenizer
        print("Инициализировали токенайзер")

        self.suggester = WordPieceTextSuggestion(
            self.word_completor, self.n_gram_model, self.tokenizer
        )

    def predict_sentence(self, sentence: str, n_words: int, n_texts: int) -> List[str]:
        return self.suggester.suggest_text(
            sentence, n_words=n_words, beam_width=n_texts
        )
