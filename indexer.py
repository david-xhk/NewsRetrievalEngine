import nltk
import string
from doctypes import Document, TokenizedDocument

stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)


def clean_(words: list[str]) -> list[str]:
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


def preprocess_(docs: list[Document]) -> list[TokenizedDocument]:
    docs_ = []
    for doc in docs:
        doc_ = TokenizedDocument(
            id=doc.id,
            title=clean_(doc.title),
            content=clean_(doc.content))
        docs_.append(doc_)
    return docs_


def words_(docs: list[TokenizedDocument]) -> list[str]:
    """Return a list of unique words across all documents."""
    words = []
    for doc in docs:
        for word in doc.title:
            words.append(word)
        for word in doc.content:
            words.append(word)
    words = set(words)
    return sorted(words)


def invert_(docs: list[TokenizedDocument],
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
    docs = preprocess_(docs)
    words = words_(docs)
    wtoi = {w: i for i, w in enumerate(words)}
    for doc in docs:
        for i, w in enumerate(doc.title):
            doc.title[i] = wtoi[w]
        for i, w in enumerate(doc.content):
            doc.content[i] = wtoi[w]
    docs_map = {doc.id: doc.to_dict() for doc in docs}
    index = invert_(docs, words, wtoi)
    return [docs_map, words, wtoi, index]


def test_main():
    input = [
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
    output = main(input)[-1]
    assert output == expected, 'expected ' + \
        str(expected) + ' from main but got ' + str(output)


test_main()

if __name__ == '__main__':
    import argparse
    import json
    from util import read_docs

    parser = argparse.ArgumentParser(
        description='Generate an inverted index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('docs_path', help='path to docs file')
    parser.add_argument('out_path', help='output path')
    parser.add_argument('--docs_type', metavar='TYPE',
                        help='type of docs file', default='csv')
    args = parser.parse_args()

    docs = read_docs(args.docs_path, args.docs_type)
    with open(args.out_path, 'w') as fp:
        json.dump(main(docs), fp)
