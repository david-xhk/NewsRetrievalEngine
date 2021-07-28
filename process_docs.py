from doctypes import Document, TokenizedDocument, LanguageModel
from util import clean_words, DataclassJSONEncoder


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


def main(
    docs: list[Document],
    do_create_inverted_index: bool = True,
    do_create_language_models: bool = True,
    smoothing_constant: int = 1,
    **kwargs,
) -> dict:
    result = {**kwargs}
    result['docs'] = preprocess_docs(docs)
    result['words'] = get_words(result['docs'])
    result['word_ids'] = create_word_ids(result['words'])
    convert_words_to_ids(result['docs'], result['word_ids'])
    if do_create_inverted_index:
        result['inverted_index'] = create_inverted_index(
            result['docs'],
            result['words'],
            result['word_ids'],
        )
    if do_create_language_models:
        result['language_models'] = create_language_models(
            result['docs'],
            result['words'],
            smoothing_constant,
        )
        result['collection_model'] = create_collection_model(
            result['language_models'],
        )
    return result


def test_main():
    input_ = [
        Document(
            id=0,
            title='breakthrough drug for schizophrenia',
            content='',
        ),
        Document(
            id=1,
            title='new schizophrenia drugs',
            content='',
        ),
        Document(
            id=2,
            title='new approach for treatment of schizophrenia',
            content='',
        ),
        Document(
            id=3,
            title='new hopes for schizophrenia patients',
            content='',
        ),
    ]
    expected = {
        'approach': [2],
        'breakthrough': [0],
        'drug': [0, 1],
        'hope': [3],
        'new': [1, 2, 3],
        'patient': [3],
        'schizophrenia': [0, 1, 2, 3],
        'treatment': [2],
    }
    output = main(input_)['inverted_index']
    assert output == expected, f'expected {expected} from main but got {output}'


test_main()

if __name__ == '__main__':
    import argparse
    import json
    import time
    from util import read_docs, get_docs_size

    parser = argparse.ArgumentParser(
        description='Process a collection of documents.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--in',
        required=True,
        help='path to input file (must be .csv or .json format)',
        metavar="PATH",
        dest='docs_path',
    )
    parser.add_argument(
        '-o',
        '--out',
        default='result.json',
        help='output path',
        metavar="PATH",
        dest='out_path',
    )
    parser.add_argument(
        '--inverted-index',
        action='store_true',
        help='if specified, create inverted index',
        dest='do_create_inverted_index',
    )
    parser.add_argument(
        '--language-models',
        action='store_true',
        help='if specified, create language models',
        dest='do_create_language_models',
    )
    parser.add_argument(
        '--smoothing-constant',
        default=0,
        type=int,
        help='smoothing constant for language models (set to 0 for no smoothing)',
        metavar=">= 0",
    )

    args = parser.parse_args()
    docs = read_docs(args.docs_path)
    docs_size = get_docs_size(args.docs_path)
    print(f'read {args.docs_path} with {len(docs)} rows @ {docs_size/1e6:.1f}MB')

    print('start indexing...')
    start = time.time()
    output = main(docs, **vars(args))
    end = time.time()
    time_taken = end - start
    speed = docs_size / 1e3 / time_taken
    time_taken = f'{time_taken // 60:.0f}m{time_taken % 60:.0f}s'
    print(f'indexing completed in {time_taken} ({speed:.1f}KB/s)')

    with open(args.out_path, 'w') as fp:
        json.dump(output, fp, cls=DataclassJSONEncoder)
