import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tokenizers import BertWordPieceTokenizer

# чтобы заново не скачивать
nltk.data.path.append(
    "/Users/andreypetukhov/Documents/Машинное-обучение/ML1-and-ML2/homework-practice-10-unsupervised/nltk_data"
)

STOP_WORDS = set(stopwords.words("english"))


def preprocess_msg(message: str) -> str:
    headers = r"(?i)(Message-ID|Date|From|To|Subject|Mime-Version|Content-Type|Content-Transfer-Encoding|X-[^\n]+|FileName):.*\n"
    emails = r"\S+@\S+\.\S+"
    urls = r"http\S+|www\.\S+"
    spaces = r"\s{2,}"  # пробелы подряд
    other = r"[^a-zA-Z\s.!?]"

    combined = f"{headers}|{emails}|{urls}|{spaces}|{other}"

    msg = re.sub(combined, " ", message)
    msg = re.sub(r"([.!?])\1+", r"\1", msg)  # повторяющиеся знаки удаляю
    msg = re.sub(r"\b's\b", " is", msg)  # заменяю 's на is
    msg = re.sub(r"\b're\b", " are", msg)
    msg = re.sub(r"\b've\b", " have", msg)
    msg = re.sub(r"\b'nt\b", " not", msg)
    msg = re.sub(r"\n+", "", msg).strip().lower()

    return msg


def word_piece_tokenize(message: str, tokenizer: BertWordPieceTokenizer) -> list:
    return tokenizer.encode(message).tokens
