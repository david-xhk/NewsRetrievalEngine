from doctypes import TokenizedDocument, LanguageModel
import functools
import math


def calculate_word_probability(
    model: LanguageModel,
    word: str,
    normalize: bool = False,
) -> float:
    """Calculate word probability given the language model of a document.

    Argument:
        model: language model for a document
        word: query word
        normalize: whether to normalize with log
    Returns:
        word probability with the given language model
    """
    p = (model.model.get(word, 0) + model.smoothing_constant) / model.total
    if normalize:
        return math.log(p)
    else:
        return p


def test_calculate_word_probability():
    input_ = (LanguageModel(id=0, total=15, smoothing_constant=1,
                            model={'a': 1, 'b': 2, 'c': 3, 'd': 4}),
              'e', False)
    expected = 0.06666666666666667
    output = calculate_word_probability(*input_)
    assert output == expected, f'expected {expected} from calculate_word_probability but got {output}'


test_calculate_word_probability()


def calculate_sentence_probability(
    model: LanguageModel,
    sentence: list[str],
    normalize: bool = False,
) -> float:
    """Calculate sentence probability given the language model of a document.

    Argument:
        model: language model for a document
        sentence: list of query words
        normalize: whether to normalize with log
    Returns:
        sentence probability with the given language model
    """
    def f(x): return calculate_word_probability(model, x)
    if normalize:
        return functools.reduce(lambda p, w: p + math.log(f(w)), sentence, 0)
    else:
        return functools.reduce(lambda p, w: p * f(w), sentence, 1)


def test_calculate_sentence_probability():
    input_ = (LanguageModel(id=0, total=15, smoothing_constant=1,
                            model={'a': 1, 'b': 2, 'c': 3, 'd': 4}),
              ['b', 'd', 'e'], False)
    expected = 0.0044444444444444444
    output = calculate_sentence_probability(*input_)
    assert output == expected, f'expected {expected} from calculate_sentence_probability but got {output}'


test_calculate_sentence_probability()


def calculate_interpolated_sentence_probability(
    model: LanguageModel,
    collection_model: LanguageModel,
    sentence: list[str],
    alpha: float = 0.75,
    normalize: bool = False,
) -> float:
    """Calculate interpolated sentence probability given the language models of a document and its collection.

    Argument:
        model: language model for a document
        colleciton_model: language model for the entire collection
        sentence: list of query words
        alpha: document-collection interpolation constant
        normalize: whether to normalize with log
    Return:
        sentence probability with the given document and collection language models
    """
    def f(x): return (calculate_word_probability(model, x) * alpha +
                      calculate_word_probability(collection_model, x) * (1 - alpha))
    if normalize:
        return functools.reduce(lambda p, w: p + math.log(f(w)), sentence, 0)
    else:
        return functools.reduce(lambda p, w: p * f(w), sentence, 1)


def test_calculate_interpolated_sentence_probability():
    input_ = (LanguageModel(id=0, total=15, smoothing_constant=1,
                            model={'a': 1, 'b': 2, 'c': 3, 'd': 4}),
              LanguageModel(id=-1, total=25, smoothing_constant=2,
                            model={'a': 1, 'b': 2, 'c': 5, 'd': 4, 'e': 3}),
              ['b', 'd', 'e'], 0.75, False)
    expected = 0.005890000000000001
    output = calculate_interpolated_sentence_probability(*input_)
    assert output == expected, f'expected {expected} from calculate_interpolated_sentence_probability but got {output}'


test_calculate_interpolated_sentence_probability()


def calculate_tf(doc: TokenizedDocument) -> dict[int, int]:
    """Calculate term frequency for all words in a document.

    Argument:
        doc: tokenized document
    Return:
        dict of words to the number of times they occur in the document
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
    assert output == expected, f'expected {expected} from calculate_tf but got {output}'


test_calculate_tf()


def calculate_df(docs: list[TokenizedDocument]) -> dict[int, int]:
    """Calculate document frequency for all words in all documents.

    Argument:
        docs: list of tokenized documents
    Returns:
        dict of words to the number of documents they appear in
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
    assert output == expected, f'expected {expected} from calculate_df but got {output}'


test_calculate_df()


def calculate_idf(df: dict[int, int], corpus_size: int) -> dict[int, float]:
    """Calculate inverse document frequency.

    Arguments:
        df: dict of words to their document frequencies (output of df_)
        corpus_size: number of documents in the corpus (len of docs)
    Returns:
        dict of words to their inverse document frequencies
    """
    idf = {}
    for i, freq in df.items():
        idf[i] = math.log(corpus_size / freq)
    return idf


def test_calculate_idf():
    input_ = ({0: 1, 1: 2, 2: 3, 3: 2, 4: 1}, 3)
    expected = {0: 1.0986122886681098,
                1: 0.4054651081081644,
                2: 0.0,
                3: 0.4054651081081644,
                4: 1.0986122886681098}
    output = calculate_idf(*input_)
    assert output == expected, f'expected {expected} from calculate_idf but got {output}'


test_calculate_idf()


def calculate_bm25(
    query: list[int],
    docs: list[TokenizedDocument],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple]:
    """Return the ids of the top k documents with the highest BM25 retrieval status value for the query.

    Arguments:
        query: tokenized query
        docs: list of tokenized documents which contain a word in the query
        k1: document frequency scaling factor
        b: document length scaling factor
    Returns:
        list of ids of the top k documents
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

    return sorted(rsv.items(), key=lambda kv: kv[1], reverse=True)


def test_calculate_bm25():
    input_ = ([1, 2, 4],
              [TokenizedDocument(id=0, title=[0, 1, 2], content=[]),
              TokenizedDocument(id=1, title=[1, 2, 3], content=[]),
              TokenizedDocument(id=2, title=[2, 3, 4], content=[])])
    expected = [(2, 1.0986122886681098),
                (0, 0.4054651081081644),
                (1, 0.4054651081081644)]
    output = calculate_bm25(*input_)
    assert output == expected, f'expected {expected} from calculate_bm25 but got {output}'


test_calculate_bm25()
