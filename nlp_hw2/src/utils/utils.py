import pandas as pd
import numpy as np
import nltk
import torch
import random
import os
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader

STOP_WORDS = set(stopwords.words("english"))

# беру пути из дз по мо-2

if "/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/nltk_data" not in nltk.data.path:
    nltk.data.path.append("/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/nltk_data")

if "/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/corpora" not in nltk.data.path:
    nltk.data.path.append("/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/corpora")


def text_to_words(text: str):
    return nltk.word_tokenize(text)

def set_random_seed(seed):
    #как на звуке :)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def collate_fn(batch, max_seq_len=None):
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    if max_seq_len:
        inputs = [text[:max_seq_len] for text in inputs]

    max_len = max(len(text) for text in inputs)
    padded_inputs = [text + [0] * (max_len - len(text)) for text in inputs]
    
    padded_inputs = torch.tensor(padded_inputs, dtype=torch.long)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    
    return {"input_ids": padded_inputs, "labels": labels}


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


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {"input_ids": text, "labels": label}