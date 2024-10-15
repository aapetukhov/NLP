from typing import List, Union

from ngram_lm import NGramLanguageModel
from tokenizers import BertWordPieceTokenizer
from utils import preprocess_msg
from word_completor import WordCompletor


class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(
        self, text: Union[str, List[str]], n_words=3, n_texts=1
    ) -> List[List[str]]:
        """
        Возвращает возможные варианты продолжения текста

        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений

        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """
        if isinstance(text, str):
            text = text.split()

        last_word_prefix = text[-1]
        completed_words, _ = self.word_completor.get_words_and_probs(last_word_prefix)

        if not completed_words:
            return []

        completed_word = completed_words[0]
        completed_text = text[:-1] + [completed_word]

        # со слова которое дополнил wordcompleter
        start_index = len(text) - 1

        suggestions = []
        for _ in range(n_texts):
            suggestion = completed_text
            for _ in range(n_words):
                next_words, probas = self.n_gram_model.get_next_words_and_probs(
                    suggestion
                )
                if next_words:
                    sorted_next_words = sorted(
                        zip(next_words, probas), key=lambda x: -x[1]
                    )
                    suggestion.append(sorted_next_words[0][0])
                else:
                    break
            suggestions.append(suggestion[start_index:])

        return suggestions


class WordPieceTextSuggestion(TextSuggestion):
    def __init__(
        self,
        word_completor: WordCompletor,
        n_gram_model: NGramLanguageModel,
        tokenizer: BertWordPieceTokenizer,
    ):
        """
        Инициализация модели:
        - word_completor: строит префиксное дерево
        - n_gram_model: строит n-граммную модель
        - tokenizer: WordPiece токенизатор

        corpus: список списков токенов (разделённые слова)
        n: порядок n-граммы
        tokenizer: WordPiece токенизатор
        """
        super().__init__(word_completor, n_gram_model)
        self.tokenizer = tokenizer

    def suggest_text(
        self, text: Union[str, List[str]], n_words=3, beam_width=3
    ) -> List[List[str]]:
        """
        Возвращает варианты продолжения текста с использованием Beam Search.

        text: строка или список слов – текст, который нужно дополнить
        n_words: число слов, которые дописывает n-граммная модель
        beam_width: количество путей, которые сохраняются на каждом шаге

        return: list[str] – список предложений с дополнением (не списком списков)
        """
        if isinstance(text, list):  # тут рабоатем только со строчками
            text = " ".join(text)

        text = preprocess_msg(text)

        encoded_input = self.tokenizer.encode(text)
        token_ids = encoded_input.ids
        tokens = encoded_input.tokens

        last_token = tokens[-1]
        completed_tokens, _ = self.word_completor.get_words_and_probs(
            last_token
        )  # с дополненным словом

        if not completed_tokens:
            return []

        first_token = completed_tokens[0]
        completed_ids = token_ids[:-1] + [self.tokenizer.token_to_id(first_token)]
        completed_tokens = tokens[:-1] + [first_token]  # с токеном дополненного слова

        # щапускаем бим-сёрч
        beams = [(0, completed_ids, completed_tokens)]

        for _ in range(n_words):
            new_beams = []
            for total_proba, predicted_ids, predicted_tokens in beams:
                pred_next_tokens, probas = self.n_gram_model.get_next_words_and_probs(
                    predicted_tokens
                )
                if pred_next_tokens:
                    # добавляем новые варианты
                    for next_token, proba in zip(pred_next_tokens, probas):
                        next_tokens = predicted_tokens + [next_token]
                        next_ids = predicted_ids + [
                            self.tokenizer.token_to_id(next_token)
                        ]
                        new_score = total_proba + proba
                        new_beams.append((new_score, next_ids, next_tokens))

            beams = sorted(new_beams, key=lambda x: x[0])[:beam_width]

            if _ == 1:
                pass

        suggestions = []
        ids = []  # для отладки
        for _, predicted_ids, _ in beams:
            decoded_suggestion = self.tokenizer.decode(
                predicted_ids[len(token_ids) - 1 :]
            )
            suggestions.append(decoded_suggestion)
            ids.append(predicted_ids[len(token_ids) - 1 :])

        return suggestions
