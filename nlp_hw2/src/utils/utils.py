import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# беру пути из дз по мо-2

if "/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/nltk_data" not in nltk.data.path:
    nltk.data.path.append("/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/nltk_data")

if "/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/corpora" not in nltk.data.path:
    nltk.data.path.append("/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/corpora")


def text_to_words(text: str):
    return nltk.word_tokenize(text)


class Binarizer:
    def __init__(self, dataset: pd.DataFrame, labels_col: str = "labels"):
        self.labels_col = labels_col
        
        items_labels = dataset[labels_col].apply(lambda x: set(x.split(", ")))
        _all_labels = set()

        for item in items_labels:
            for label in item:
                _all_labels.add(label)
        
        self._all_labels = sorted(_all_labels)

    def transform(self, dataset: pd.DataFrame, add_every_label: bool = False):
        """
        Transforms in-place
        """
        # добавляем колонку с вектором ответов
        dataset["binary_labels"] = dataset[self.labels_col].apply(lambda x: self.binary_encode(x.split(", "), self._all_labels))

        if add_every_label:
            # добавляем по колонке для каждого лейбла
            binary_columns = pd.DataFrame(dataset["binary_labels"].tolist(), columns=self._all_labels, index=dataset.index)
            dataset[self._all_labels] = binary_columns

    @staticmethod
    def binary_encode(label_list, _all_labels):
        return np.array([1 if label in label_list else 0 for label in _all_labels])
    
    @property
    def all_labels(self):
        return self._all_labels
