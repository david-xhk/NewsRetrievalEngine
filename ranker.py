from __future__ import annotations

import functools
import math
from collections import Counter

from doctypes import LanguageModel, TokenizedDocument


def calculate_precision_at_k(R: list, k: int) -> float:
    """Calculate precision @ K of a ranking.

    Arguments:
        R: relevance mapping of a ranking (nonzero is relevant)
        k: position to calculate at
    Returns:
        Precision @ K of the ranking
    """
    return sum(R[:k]) / k


def calculate_average_precision(R: list) -> float:
    """Calculate average precision of a ranking.

    Arguments:
        R: relevance mapping of a ranking (nonzero is relevant)
    Returns:
        Average precision of the ranking
    """
    p = []
    for i, r in enumerate(R):
        if r > 0:
            p.append(calculate_precision_at_k(R, i + 1))
    return sum(p) / len(p) if p else 0.0


def calculate_mean_average_precision(Rs: list[list]) -> float:
    """Calculate mean average precision of some rankings.

    Arguments:
        Rs: relevance mappings of some rankings (nonzero is relevant)
    Returns:
        Mean average precision of the rankings
    """
    p = []
    for R in Rs:
        p.append(calculate_average_precision(R))
    return sum(p) / len(p)


def calculate_dcg_at_k(R: list, k: int) -> float:
    """Calculate discounted cumulative gain @ K of a ranking.

    Arguments:
        R: relevance mapping of a ranking (nonzero is relevant)
        k: position to calculate at
    Returns:
        DCG @ K of the ranking
    """
    p = R[0]
    for i in range(1, min(len(R), k)):
        p += R[i] / math.log2(i + 1)
    return p


def calculate_ndcg_at_k(R: list, k: int) -> float:
    """Calculate normalized DCG @ K of a ranking.

    Args:
        R: relevance mapping of a ranking (nonzero is relevant)
        k: position to calculate at
    Returns:
        NDCG @ K of the ranking
    """
    S = sorted(R, reverse=True)
    return calculate_dcg_at_k(R, k) / (calculate_dcg_at_k(S, k) or 1e-23)


def calculate_mean_reciprocal_rank(Rs: list) -> float:
    """Calculate mean reciprocal rank of some rankings.

    Arguments:
        Rs: relevance mappings of some rankings (nonzero is relevant)
    Returns:
        Mean reciprocal rank of the rankings
    """
    p = []
    for R in Rs:
        s = 0.0
        for i, r in enumerate(R):
            if r > 0:
                s = 1 / (i + 1)
                break
        p.append(s)
    return sum(p) / len(p)


def calculate_jaccard_similarity(doc1: list, doc2: list) -> float:
    """Calculate Jaccard similarity score between two documents.

    Arguments:
        doc1: document 1
        doc2: document 2
    Returns:
        Jaccard similarity score between doc1 and doc2
    """
    A = set(doc1)
    B = set(doc2)
    return len(A.intersection(B)) / len(A.union(B))


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
    p = (model.counter[word] + model.smoothing_constant) / model.total
    if normalize:
        return math.log(p)
    else:
        return p


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


def calculate_tf(doc: TokenizedDocument[str]) -> Counter[str, int]:
    """Calculate term frequency for all words in a document.

    Argument:
        doc: tokenized document
    Return:
        dict of words to the number of times they occur in the document
    """
    return Counter(doc.title + doc.content)


def calculate_df(docs: list[TokenizedDocument[str]]) -> Counter[str, int]:
    """Calculate document frequency for all words in all documents.

    Argument:
        docs: list of tokenized documents
    Returns:
        dict of words to the number of documents they appear in
    """
    counter = Counter()
    for doc in docs:
        counter.update(set(doc.title + doc.content))
    return counter


def calculate_idf(df: Counter[str, int], corpus_size: int) -> dict[str, float]:
    """Calculate inverse document frequency.

    Arguments:
        df: dict of words to their document frequencies (output of df_)
        corpus_size: number of documents in the corpus (len of docs)
    Returns:
        dict of words to their inverse document frequencies
    """
    idf = {}
    for word, freq in df.items():
        idf[word] = math.log(corpus_size / freq)
    return idf


def calculate_bm25(
    query: list[str],
    docs: list[TokenizedDocument[str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    """Return the ids of the top k documents with the highest BM25 scores for the query.

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

    scores = {}  # maps document id to score
    for doc in docs:
        L_d = len(doc.title + doc.content)
        rsv_d = 0.0
        tf_d = calculate_tf(doc)
        for word in query:
            if word not in tf_d:
                continue
            idf_t = idf[word]
            tf_td = tf_d[word]
            rsv_td = idf_t * (((k1 + 1) * tf_td) /
                              ((k1 * ((1 - b) + b * (L_d / L_ave)) + tf_td)))
            rsv_d += rsv_td
        scores[doc.id] = rsv_d

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
