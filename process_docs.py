from doctypes import Document, TokenizedDocument, LanguageModel
from util import read_docs, get_docs_size, clean_words, DataclassJSONEncoder
import json
import time


def preprocess_docs(docs: list[Document]) -> list[TokenizedDocument]:
    docs_ = []
    for doc in docs:
        doc_ = TokenizedDocument(
            id=doc.id,
            title=clean_words(doc.title),
            content=clean_words(doc.content),
        )
        docs_.append(doc_)
    return docs_


def convert_words_to_ids(
    docs: list[TokenizedDocument],
    word_ids: dict[str, int],
) -> list[TokenizedDocument]:
    for doc in docs:
        for sentence in [doc.title, doc.content]:
            for i, w in enumerate(sentence):
                sentence[i] = word_ids[w]


def create_language_models(
    docs: list[TokenizedDocument],
    words: list[str],
    smoothing_constant: int = 1,
) -> list[LanguageModel]:
    vocab_size = len(words)
    models = []
    for doc in docs:
        model_ = {}
        total = vocab_size * smoothing_constant
        for sentence in [doc.title, doc.content]:
            for word_id in sentence:
                word = words[word_id]
                model_[word] = model_.get(word, 0) + 1
                total += 1
        model = LanguageModel(
            id=doc.id,
            model=model_,
            total=total,
            smoothing_constant=smoothing_constant,
        )
        models.append(model)
    return models


def create_collection_model(
    models: list[LanguageModel],
) -> LanguageModel:
    model_ = {}
    total = 0
    smoothing_constant = 0
    for model in models:
        for word, count in model.model.items():
            model_[word] = model_.get(word, 0) + count
        total += model.total
        smoothing_constant += model.smoothing_constant
    collection_model = LanguageModel(
        id=-1,
        model=model_,
        total=total,
        smoothing_constant=smoothing_constant,
    )
    return collection_model


def get_words(docs: list[TokenizedDocument]) -> list[str]:
    """Return a list of unique words across all documents."""
    words = []
    for doc in docs:
        for sentence in [doc.title, doc.content]:
            for word in sentence:
                words.append(word)
    words = set(words)
    return sorted(words)


def create_word_ids(words: list[str]) -> dict[str, int]:
    return {word: id for id, word in enumerate(words)}


def create_inverted_index(
    docs: list[TokenizedDocument],
    words: list[str],
    word_ids: dict[str, int],
) -> dict[str, list[int]]:  # Mapping of words to postings lists
    index = {}
    for word in words:
        postings = []
        for doc in docs:
            word_id = word_ids[word]
            if word_id in doc.title or word_id in doc.content:
                postings.append(doc.id)
        index[word] = postings
    return index


def test_create_inverted_index():
    input_ = [
        [
            TokenizedDocument(id=0, title=[0, 1, 2, 3], content=[]),
            TokenizedDocument(id=1, title=[4, 3, 1, 2], content=[]),
            TokenizedDocument(id=2, title=[4, 1, 0, 3], content=[]),
            TokenizedDocument(id=3, title=[4, 2, 3, 0], content=[]),
        ],
        ['a', 'b', 'c', 'd', 'e'],
        {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
    ]
    expected = {
        'a': [0, 2, 3],
        'b': [0, 1, 2],
        'c': [0, 1, 3],
        'd': [0, 1, 2, 3],
        'e': [1, 2, 3],
    }
    output = create_inverted_index(*input_)
    assert output == expected, f'expected {expected} from create_inverted_index but got {output}'


test_create_inverted_index()


def main(
    docs_path: str = 'test_data.csv',
    out_path: str = 'test_data_processed.json',
    do_create_inverted_index: bool = True,
    do_create_language_models: bool = True,
    smoothing_constant: int = 1,
):
    """Process a collection of documents.

    Arguments:
        docs_path: path to input file (must be .csv or .json format)
        out_path: output path (must be .json format)
        do_create_inverted_index: if set to true, create inverted index
        do_create_language_models: if set to true, create language models
        smoothing_constant: smoothing constant for language models (set to 0 for no smoothing)
    """
    docs = read_docs(docs_path)
    docs_size = get_docs_size(docs_path)
    print(f'read {docs_path} with {len(docs)} rows @ {docs_size/1e6:.1f}MB')

    print('start indexing...')
    start = time.time()

    docs = preprocess_docs(docs)
    words = get_words(docs)
    word_ids = create_word_ids(words)
    convert_words_to_ids(docs, word_ids)

    result = {'docs': docs, 'words': words, 'word_ids': word_ids}
    if do_create_inverted_index:
        inverted_index = create_inverted_index(docs, words, word_ids)
        result['inverted_index'] = inverted_index
    if do_create_language_models:
        models = create_language_models(docs, words, smoothing_constant)
        result['language_models'] = models
        result['collection_model'] = create_collection_model(models)

    end = time.time()
    time_taken = end - start
    speed = docs_size / 1e3 / time_taken
    time_taken = f'{time_taken // 60:.0f}m{time_taken % 60:.0f}s'
    print(f'indexing completed in {time_taken} ({speed:.1f}KB/s)')

    with open(out_path, 'w') as fp:
        json.dump(result, fp, cls=DataclassJSONEncoder)
    print(f'wrote result to {out_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process a collection of documents.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--in', default='test_data.csv',
        help='path to input file (must be .csv or .json format)',
        metavar='PATH', dest='docs_path')
    parser.add_argument(
        '-o', '--out', default='test_data_processed.json',
        help='output path (must be .json format)',
        metavar='PATH', dest='out_path')
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
