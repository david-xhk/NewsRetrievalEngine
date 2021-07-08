import nltk
import string
import pandas as pd
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


def get_docs(docs_path: str, docs_type: str) -> pd.DataFrame:
    reader = getattr(pd, f'read_{docs_type}')
    df = reader(docs_path)
    return df


def read_docs(docs_path: str, docs_type: str) -> list[Document]:
    df = get_docs(docs_path, docs_type)
    docs = df.apply(row_to_doc_adapter, axis=1)
    return docs.tolist()


def get_doc(docid: int, docs_path: str, docs_type: str) -> Document:
    df = get_docs(docs_path, docs_type)
    res = df[df['ID'] == docid]
    if not res.empty:
        return row_to_doc_adapter(res.iloc[0])


def test_get_doc():
    input_ = (315201, 'test_data.csv', 'csv')
    expected = 315201
    output = get_doc(*input_)
    assert output.id == expected, 'expected ' + str(
        expected) + ' from get_doc but got ' + str(output)


test_get_doc()


def get_docs_size(docs_path: str, docs_type: str) -> int:
    df = get_docs(docs_path, docs_type)
    return df.memory_usage(deep=True).sum()
