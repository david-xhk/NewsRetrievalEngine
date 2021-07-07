from doctypes import TokenizedDocument
from math import log


def calculate_tf(doc: TokenizedDocument) -> dict[int, int]:
    """Calculate term frequency for all words in a document.

    Argument:
        - doc: tokenized document
    Returns:
        - dict of words to the number of times they occur in the document
    """
    tf = {}
    for i in doc.title + doc.content:
        if i not in tf:
            tf[i] = 0
        tf[i] += 1
    return tf


def test_calculate_tf():
    input_ = TokenizedDocument(id=0, title=[0, 1, 2, 3], content=[])
    expected = {0: 1, 1: 1, 2: 1, 3: 1}
    output = calculate_tf(input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from calculate_tf but got ' + str(output)


test_calculate_tf()


def calculate_df(docs: list[TokenizedDocument]) -> dict[int, int]:
    """Calculate document frequency for all words in all documents.

    Argument:
        - docs: list of tokenized documents
    Returns:
        - dict of words to the number of documents they appear in
    """
    df = {}
    for doc in docs:
        for i in set(doc.title + doc.content):
            if i not in df:
                df[i] = 0
            df[i] += 1
    return df


def test_calculate_df():
    input_ = [TokenizedDocument(id=0, title=[0, 1, 2], content=[]),
              TokenizedDocument(id=1, title=[1, 2, 3], content=[]),
              TokenizedDocument(id=2, title=[2, 3, 4], content=[])]
    expected = {0: 1, 1: 2, 2: 3, 3: 2, 4: 1}
    output = calculate_df(input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from calculate_df but got ' + str(output)


test_calculate_df()


def calculate_idf(df: dict[int, int], corpus_size: int) -> dict[int, float]:
    """Calculate inverse document frequency.

    Arguments:
        - df: dict of words to their document frequencies (output of df_)
        - corpus_size: number of documents in the corpus (len of docs)
    Returns:
        - dict of words to their inverse document frequencies
    """
    idf = {}
    for i, freq in df.items():
        idf[i] = log(corpus_size / freq)
    return idf


def test_calculate_idf():
    input_ = ({0: 1, 1: 2, 2: 3, 3: 2, 4: 1}, 3)
    expected = {0: 1.0986122886681098,
                1: 0.4054651081081644,
                2: 0.0,
                3: 0.4054651081081644,
                4: 1.0986122886681098}
    output = calculate_idf(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from calculate_idf but got ' + str(output)


test_calculate_idf()


def calculate_bm25(
        query: list[int],
        docs: list[TokenizedDocument],
        topk: int = 10,
        k1: float = 1.2,
        b: float = 0.75) -> list[tuple]:
    """Return the ids of the top k documents with the highest BM25 retrieval status value for the query.

    Arguments:
        - query: tokenized query
        - docs: list of tokenized documents which contain a word in the query
        - topk: number of results to return
        - k1: document frequency scaling factor
        - b: document length scaling factor
    Returns:
        - list of ids of the top k documents
    """
    N = len(docs)
    L_ave = sum(len(doc.title + doc.content)
                for doc in docs) / N
    df = calculate_df(docs)
    idf = calculate_idf(df, N)

    rsv = {}  # maps document id to score
    for doc in docs:
        L_d = len(doc.title + doc.content)
        rsv_d = 0.0
        tf_d = calculate_tf(doc)
        for i in query:
            if i not in tf_d:
                continue
            idf_t = idf[i]
            tf_td = tf_d[i]
            rsv_td = idf_t * (((k1 + 1) * tf_td) /
                              ((k1 * ((1 - b) + b * (L_d / L_ave)) + tf_td)))
            rsv_d += rsv_td
        rsv[doc.id] = rsv_d

    rank = sorted(rsv.items(), key=lambda kv: kv[1], reverse=True)
    return rank[:topk]


def test_calculate_bm25():
    input_ = ([1, 2, 4],
              [TokenizedDocument(id=0, title=[0, 1, 2], content=[]),
               TokenizedDocument(id=1, title=[1, 2, 3], content=[]),
               TokenizedDocument(id=2, title=[2, 3, 4], content=[])])
    expected = [(2, 1.0986122886681098),
                (0, 0.4054651081081644),
                (1, 0.4054651081081644)]
    output = calculate_bm25(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from calculate_bm25 but got ' + str(output)


test_calculate_bm25()
