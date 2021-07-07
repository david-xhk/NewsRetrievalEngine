from doctypes import TokenizedDocument
from math import log


def tf_(doc: TokenizedDocument, words: list[str]) -> dict[str, int]:
    """Calculate term frequency for all words in a document.

    Argument:
        - doc: tokenized document
        - words: word list
    Returns:
        - dict of words to the number of times they occur in the document
    """
    tf = {}
    for idx in doc.title + doc.content:
        word = words[idx]
        if word not in tf:
            tf[word] = 0
        tf[word] += 1
    return tf


def test_tf_():
    input_ = (TokenizedDocument(id=0, title=[0, 1, 2, 3], content=[]),
              ['a', 'b', 'c', 'd'])
    expected = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
    output = tf_(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from tf_ but got ' + str(output)


test_tf_()


def df_(docs: list[TokenizedDocument], words: list[str]) -> dict[str, int]:
    """Calculate document frequency for all words in all documents.

    Argument:
        - docs: list of tokenized documents
        - words: word list
    Returns:
        - dict of words to the number of documents they appear in
    """
    df = {}
    for doc in docs:
        for idx in set(doc.title + doc.content):
            word = words[idx]
            if word not in df:
                df[word] = 0
            df[word] += 1
    return df


def test_df_():
    input_ = ([TokenizedDocument(id=0, title=[0, 1, 2], content=[]),
               TokenizedDocument(id=1, title=[1, 2, 3], content=[]),
               TokenizedDocument(id=2, title=[2, 3, 4], content=[])],
              ['a', 'b', 'c', 'd', 'e'])
    expected = {'a': 1, 'b': 2, 'c': 3, 'd': 2, 'e': 1}
    output = df_(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from df_ but got ' + str(output)


test_df_()


def idf_(df: dict[str, int], corpus_size: int) -> dict[str, float]:
    """Calculate inverse document frequency.

    Arguments:
        - df: dict of words to their document frequencies (output of df_)
        - corpus_size: number of documents in the corpus (len of docs)
    Returns:
        - dict of words to their inverse document frequencies
    """
    idf = {}
    for word, freq in df.items():
        idf[word] = round(log(corpus_size / freq), 2)
    return idf


def test_idf_():
    input_ = ({'a': 1, 'b': 2, 'c': 3, 'd': 2, 'e': 1}, 3)
    expected = {'a': 1.1, 'b': 0.41, 'c': 0.0, 'd': 0.41, 'e': 1.1}
    output = idf_(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from idf_ but got ' + str(output)


test_idf_()


def bm25_(
        query: list[int],
        doc: TokenizedDocument,
        docs: list[TokenizedDocument],
        words: list[str],
        k1: float = 1.2,
        b: float = 0.75) -> float:
    """Calculate the BM25 retrieval status value for a query and document.

    Arguments:
        - query: tokenized query
        - doc: tokenized document
        - docs: list of tokenized documents which contain a word in the query
        - words: word list
        - k1: document frequency scaling factor
        - b: document length scaling factor
    Returns:
        - BM25 retrieval status value for the query and document
    """
    rsv_d = 0.0
    N = len(docs)
    L_d = len(doc.title + doc.content)
    L_ave = sum(len(doc.title + doc.content)
                for doc in docs) / N  # calculate average document length
    tf_d = tf_(doc, words)
    df = df_(docs, words)
    idf = idf_(df, N)
    for t in query:
        if t not in tf_d:
            continue
        idf_t = idf[t]
        tf_td = tf_d[t]
        rsv_td = idf_t * (((k1 + 1) * tf_td) /
                          ((k1 * ((1 - b) + b * (L_d / L_ave)) + tf_td)))
        rsv_d += rsv_td
    return round(rsv_d, 2)


def test_bm25_():
    input_ = ([1, 2, 4],
              TokenizedDocument(id=1, title=[1, 2, 3], content=[]),
              [TokenizedDocument(id=0, title=[0, 1, 2], content=[]),
               TokenizedDocument(id=1, title=[1, 2, 3], content=[]),
               TokenizedDocument(id=2, title=[2, 3, 4], content=[])],
              ['a', 'b', 'c', 'd', 'e'])
    expected = 0.41
    output = bm25_(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from bm25_ but got ' + str(output)


test_bm25_()
