import nltk
import string
import pandas as pd
import dataclasses
import json
import os
from doctypes import Document


def clean_words(words: str) -> list[str]:
    stemmer = nltk.stem.PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    words_ = []
    for word in nltk.tokenize.word_tokenize(words):
        # removes non-ascii characters
        word = word.encode('ascii', 'ignore').decode()
        if (word and not all(letter in punctuation for letter in word)):
            word = word.lower()
            if word not in stopwords:
                word = stemmer.stem(word)
                words_.append(word)
    return words_


def test_clean_words():
    input_ = '3 policemen killed as Islamic militants attack town'
    expected = ['3', 'policemen', 'kill', 'islam', 'milit', 'attack', 'town']
    output = clean_words(input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from clean_words but got ' + str(output)


test_clean_words()


def row_to_doc_adapter(row: pd.Series) -> Document:
    return Document(id=row.ID,
                    title=row.title,
                    content=row.content)


def read_df(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path)
    if ext == '.csv':
        return pd.read_csv(path)
    elif ext == '.json':
        return pd.read_json(path)
    else:
        err_msg = f"Invalid file extension '{ext}'. Must be '.csv' or '.json'."
        raise ValueError(err_msg)


def read_docs(docs_path: str) -> list[Document]:
    df = read_df(docs_path)
    docs = df.apply(row_to_doc_adapter, axis=1)
    return docs.tolist()


def get_doc(doc_id: int, docs_path: str) -> Document:
    df = read_df(docs_path)
    res = df[df['ID'] == doc_id]
    if not res.empty:
        return row_to_doc_adapter(res.iloc[0])


def test_get_doc():
    input_ = (315201, 'test_data.csv')
    expected = 315201
    output = get_doc(*input_)
    assert output.id == expected, 'expected ' + str(
        expected) + ' from get_doc but got ' + str(output)


test_get_doc()


def get_docs_size(docs_path: str) -> int:
    df = read_df(docs_path)
    return df.memory_usage(deep=True).sum()


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
