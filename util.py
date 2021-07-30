from __future__ import annotations

import dataclasses
import functools
import json
import os
import string
from timeit import default_timer as timer
from typing import Literal

import nltk
import pandas as pd
import torch
from torchtext.vocab import Vocab

from doctypes import Document, TokenizedDocument
from ranker import calculate_mean_reciprocal_rank


def fmt_secs(s: float):
    if s == 0:
        return '0s'
    ut = []
    for u, t in [('d', 86400), ('h', 3600), ('m', 60), ('s', 1)]:
        q, s = divmod(s, t)
        if q:
            ut.append(f'{int(q)}{u}')
    if any(u[-1] in 'dhm' for u in ut):
        return ''.join(ut[:2])
    s += q
    for u in ('s', 'ms', 'µs', 'ns', 'ps'):
        if s < 0.01:
            s *= 1000
        else:
            break
    return f'{round(s, 2)}{u}'


def timed(fn, args, kwargs=None):
    start_time = timer()
    output = fn(*args, **(kwargs or {}))
    end_time = timer()
    time_taken = end_time - start_time
    return output, time_taken


def test(fn, expected, *args, **kwargs):
    output, time_taken = timed(fn, args, kwargs)
    if callable(expected):
        output, expected = expected(output)
    assert output == expected, f'expected {expected} from {fn.__name__} but got {output}'
    print(f'{fn.__name__} test passed ({fmt_secs(time_taken)})')


def test_search(search_fn, *args,
                test_data_path: str = 'files/test_queries.csv',
                processed_data_path: str = 'files/test_data_processed.pickle',
                raw_data_path: str = 'files/test_data.csv',
                verbose: bool = False,
                **kwargs):
    total_time = []
    Rs = []
    df = pd.read_csv(test_data_path)
    if verbose:
        print(f'{search_fn.__name__} test start')
    for i, expected, query in df.itertuples():
        args_ = (query, processed_data_path) + args
        output, time_taken = timed(search_fn, args_, kwargs)
        total_time.append(time_taken)
        R = [int(doc_id == expected) for doc_id, _ in output]
        Rs.append(R)
        if verbose:
            print(f'  {i+1:2}. {query=}, {expected=} ({fmt_secs(time_taken)})')
            for j, (doc_id, score) in enumerate(output):
                doc = get_doc(doc_id, raw_data_path)
                id, title = doc.id, doc.title
                print(f'    {j+1:2}) {id=:7}, {score=:5.2f}, {title=}')
    time_taken = sum(total_time)
    MRR = calculate_mean_reciprocal_rank(Rs)
    latency = len(total_time) / time_taken
    print(f'{search_fn.__name__} test complete ({fmt_secs(time_taken)}): '
          f'{i+1} queries, {MRR=:.2f}, {latency=:.2f} queries/s')


def clean_words(words: str) -> list[str]:
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    cleaned = []
    for word in nltk.tokenize.word_tokenize(words):
        # removes non-ascii characters
        word = word.encode('ascii', 'ignore').decode()
        if (word and not all(letter in punctuation for letter in word)):
            word = word.lower()
            if word not in stopwords:
                cleaned.append(word)
    return cleaned


def row_to_doc_adapter(row: pd.Series) -> Document:
    return Document(id=row.ID,
                    title=row.title,
                    content=row.content)


@functools.lru_cache
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


def get_docs_size(docs_path: str) -> int:
    df = read_df(docs_path)
    return df.memory_usage(deep=True).sum()


def convert_stoi(sentence: list[str], vocab: Vocab) -> list[int]:
    for i, w in enumerate(sentence):
        sentence[i] = vocab[w]
    return sentence


def convert_itos(sentence: list[int], vocab: Vocab) -> list[str]:
    for i, w in enumerate(sentence):
        sentence[i] = vocab.itos[w]
    return sentence


def replace_unknown_words(sentence: list[str], vocab: Vocab) -> list[str]:
    for i, w in enumerate(sentence):
        sentence[i] = vocab.itos[vocab[w]]
    return sentence


def pad_sentence(
    sentence: list[int],
    vocab: Vocab,
    max_len: int,
    trim_end: bool = True,
    pad_end: bool = False,
) -> list[int]:
    sentence_len = len(sentence)
    pad_len = max(max_len - sentence_len, 0)
    padding = [vocab.stoi['<pad>']] * pad_len
    sentence_len = min(max_len, sentence_len)
    sentence = sentence[:sentence_len] if trim_end else sentence[-sentence_len:]
    return sentence + padding if pad_end else padding + sentence


def query_pipeline(
    query: str,
    vocab: Vocab,
    max_len: int | None,
    to: Literal['tensor', 'str'] = 'tensor',
) -> torch.Tensor:
    query = clean_words(query)
    query = convert_stoi(query, vocab)
    if max_len is not None:
        query = pad_sentence(query, vocab, max_len, trim_end=False)
    if to == 'tensor':
        return torch.tensor(query, dtype=torch.int64)
    else:
        return convert_itos(query, vocab)


def doc_pipeline(
    doc: TokenizedDocument[int],
    vocab: Vocab,
    max_len: int,
) -> torch.Tensor:
    doc = pad_sentence(doc.title + doc.content, vocab, max_len, trim_end=True)
    return torch.tensor(doc, dtype=torch.int64)


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
