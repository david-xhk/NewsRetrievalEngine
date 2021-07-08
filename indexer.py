from doctypes import Document, TokenizedDocument
from util import clean_words


def preprocess_docs(docs: list[Document]) -> list[TokenizedDocument]:
    docs_ = []
    for doc in docs:
        doc_ = TokenizedDocument(
            id=doc.id,
            title=clean_words(doc.title),
            content=clean_words(doc.content))
        docs_.append(doc_)
    return docs_


def get_words(docs: list[TokenizedDocument]) -> list[str]:
    """Return a list of unique words across the documents."""
    words = []
    for doc in docs:
        for word in doc.title:
            words.append(word)
        for word in doc.content:
            words.append(word)
    words = set(words)
    return sorted(words)


def create_inverted_index(
        docs: list[TokenizedDocument],
        words: list[str],
        wtoi: dict[str, int]) -> dict[str, list[int]]:
    """Return the inverted index for all words in the documents."""
    index = {}
    for word in words:
        postings = []
        for doc in docs:
            idx = wtoi[word]
            if idx in doc.title or idx in doc.content:
                postings.append(doc.id)
        index[word] = postings
    return index


def main(docs: list[Document]) -> list:
    docs = preprocess_docs(docs)
    words = get_words(docs)
    wtoi = {w: i for i, w in enumerate(words)}
    for doc in docs:
        for i, w in enumerate(doc.title):
            doc.title[i] = wtoi[w]
        for i, w in enumerate(doc.content):
            doc.content[i] = wtoi[w]
    index = create_inverted_index(docs, words, wtoi)
    docs = [doc.to_dict() for doc in docs]
    return [docs, words, wtoi, index]


def test_main():
    input_ = [
        Document(
            id=0, title='breakthrough drug for schizophrenia',
            content=''),
        Document(
            id=1, title='new schizophrenia drugs',
            content=''),
        Document(
            id=2, title='new approach for treatment of schizophrenia',
            content=''),
        Document(
            id=3, title='new hopes for schizophrenia patients',
            content='')]
    expected = {
        'approach': [2],
        'breakthrough': [0],
        'drug': [0, 1],
        'hope': [3],
        'new': [1, 2, 3],
        'patient': [3],
        'schizophrenia': [0, 1, 2, 3],
        'treatment': [2]}
    output = main(input_)[-1]
    assert output == expected, 'expected ' + \
        str(expected) + ' from main but got ' + str(output)


test_main()

if __name__ == '__main__':
    import argparse
    import json
    import time
    from util import read_docs, get_docs_size

    parser = argparse.ArgumentParser(
        description='Generate an inverted index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('docs_path', help='path to docs file')
    parser.add_argument('out_path', help='output path')
    parser.add_argument('--docs_type', metavar='TYPE',
                        help='type of docs file', default='csv')
    args = parser.parse_args()

    docs = read_docs(args.docs_path, args.docs_type)
    docs_size = get_docs_size(args.docs_path, args.docs_type)
    print(f"read {args.docs_path} with {len(docs)} rows @ {docs_size/1e6:.1f}MB")

    print("start indexing...")
    start = time.time()
    output = main(docs)
    end = time.time()
    time_taken = end - start
    speed = docs_size / 1e3 / time_taken
    time_taken = f"{time_taken // 60:.0f}m{time_taken % 60:.0f}s"
    print(f"indexing completed in {time_taken} ({speed:.1f}KB/s)")

    with open(args.out_path, 'w') as fp:
        json.dump(output, fp)
