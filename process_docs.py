from __future__ import annotations

import pickle
import time
from collections import Counter

from torchtext.vocab import Vocab

from doctypes import Document, LanguageModel, TokenizedDocument
from util import clean_words, convert_stoi, get_docs_size, read_docs


def preprocess_docs(docs: list[Document]) -> list[TokenizedDocument[str]]:
    docs_ = []
    for doc in docs:
        doc_ = TokenizedDocument(id=doc.id, title=clean_words(doc.title),
                                 content=clean_words(doc.content))
        docs_.append(doc_)
    return docs_


def create_language_models(
    docs: list[TokenizedDocument[str]],
    vocab: Vocab,
    smoothing_constant: int = 1,
) -> list[LanguageModel]:
    models = []
    for doc in docs:
        counter = Counter(doc.title + doc.content)
        total = smoothing_constant * len(vocab) + sum(counter.values())
        model = LanguageModel(id=doc.id, counter=counter, total=total,
                              smoothing_constant=smoothing_constant)
        models.append(model)
    return models


def create_collection_model(models: list[LanguageModel]) -> LanguageModel:
    counter = Counter()
    total = 0
    smoothing_constant = 0
    for model in models:
        counter += model.counter
        total += model.total
        smoothing_constant += model.smoothing_constant
    return LanguageModel(id=None, counter=counter, total=total,
                         smoothing_constant=smoothing_constant)


def create_vocab(docs: list[TokenizedDocument[str]]) -> Vocab:
    counter = Counter()
    for doc in docs:
        counter.update(doc.title)
        counter.update(doc.content)
    return Vocab(counter)


def create_inverted_index(
    docs: list[TokenizedDocument[str]],
    vocab: Vocab,
) -> dict[str, list[int]]:  # Mapping of words to lists of document ids
    index = {word: set() for word in vocab.stoi}
    for doc in docs:
        for word in doc.title + doc.content:
            index[word].add(doc.id)
    for word in index:
        index[word] = sorted(index[word])
    return index


def main(
    input_path: str = 'files/test_data.csv',
    output_path: str = 'files/test_data_processed.pickle',
    do_create_inverted_index: bool = True,
    do_create_language_models: bool = True,
    smoothing_constant: int = 1,
):
    """Process a collection of documents.

    Arguments:
        input_path: path to load input file (must be .csv or .json format)
        output_path: path to save output
        do_create_inverted_index: if set to true, create inverted index
        do_create_language_models: if set to true, create language models
        smoothing_constant: smoothing constant for language models (set to 0 for no smoothing)
    """
    docs = read_docs(input_path)
    docs_size = get_docs_size(input_path)
    print(f'read {input_path} with {len(docs)} rows @ {docs_size/1e6:.1f}MB')

    print('start indexing...')
    start = time.time()

    docs = preprocess_docs(docs)
    vocab = create_vocab(docs)
    result = {'docs': docs, 'vocab': vocab}
    if do_create_inverted_index:
        inverted_index = create_inverted_index(docs, vocab)
        result['inverted_index'] = inverted_index
    if do_create_language_models:
        models = create_language_models(docs, vocab, smoothing_constant)
        result['language_models'] = models
        result['collection_model'] = create_collection_model(models)
    for doc in docs:
        convert_stoi(doc.title, vocab)
        convert_stoi(doc.content, vocab)

    end = time.time()
    time_taken = end - start
    speed = docs_size / 1e3 / time_taken
    time_taken = f'{time_taken // 60:.0f}m{time_taken % 60:.0f}s'
    print(f'indexing completed in {time_taken} ({speed:.1f}KB/s)')

    save_processed_data(result, output_path)
    print(f'result saved at {output_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process a collection of documents.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--in', default='files/test_data.csv',
        help='path to load input file (must be .csv or .json format)',
        metavar='PATH', dest='input_path')
    parser.add_argument(
        '-o', '--out', default='files/test_data_processed.pickle',
        help='path to save output',
        metavar='PATH', dest='output_path')
    parser.add_argument(
        '-j', '--inverted-index', action='store_true',
        help='if set to true, create inverted index',
        dest='do_create_inverted_index')
    parser.add_argument(
        '-m', '--language-model', action='store_true',
        help='if set to true, create language models',
        dest='do_create_language_models')
    parser.add_argument(
        '-k', '--smoothing', default=0, type=int,
        help='smoothing constant for language models (set to 0 for no smoothing)',
        metavar='K', dest='smoothing_constant')

    args = parser.parse_args()
    main(**vars(args))
