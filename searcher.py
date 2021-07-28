from doctypes import Document, LanguageModel
from ranker import calculate_bm25, calculate_interpolated_sentence_probability
from util import clean_words, get_doc
import json


def bm25_search(
    query: str,
    topk: int,
    data_path: str,
) -> list[tuple[int, int]]:
    """Return the top k document ids and scores using BM25.

    Arguments:
        query: query string
        topk: number of results to return
        data_path: path to processed data
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Load the processed data
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    docs = [Document(**doc) for doc in data['docs']]
    docs_map = {doc.id: doc for doc in docs}
    words = data['words']
    word_ids = data['word_ids']
    inverted_index = data['inverted_index']

    # Clean and tokenize the query
    query = [word for word in clean_words(query) if word in words]
    if not query:
        return []

    # Get the ids of all documents with a word in the query
    doc_ids = set(doc_id for word in query for doc_id in inverted_index[word])

    # Get the respective documents
    hits = [docs_map[doc_id] for doc_id in doc_ids]

    # Convert query to word ids
    query = [word_ids[word] for word in query]

    # Call the BM25 algorithm and return the results
    return calculate_bm25(query, hits, topk)


def test_bm25_search():
    import pandas as pd
    import time
    test_data = pd.read_csv('test_queries.csv')
    total_ms = 0
    for i, expected, query in test_data.itertuples():
        print(f'{i+1:2}. {query=}, {expected=}')
        input_ = (query, 3, 'test_index.json')
        start = time.time()
        output = bm25_search(*input_)
        end = time.time()
        time_taken = (end - start) * 1000
        for j, (doc_id, score) in enumerate(output):
            doc = get_doc(doc_id, 'test_data.csv')
            print(f'   {j+1:2}) {doc.id=:7}, {score=:5.2f}, {doc.title=}')
        print(f'    {time_taken=:.0f}ms\n')
        total_ms += time_taken
    avg_time_taken = total_ms / (i + 1)
    latency = 1000 / avg_time_taken
    print(f'in {i+1} queries: {avg_time_taken=:.0f}ms, {latency=:.2f} queries/s')


# test_bm25_search()


def qlm_search(
    query: str,
    topk: int,
    data_path: str,
    alpha: int = 0.75,
    normalize: bool = False,
) -> list[tuple[int, int]]:
    """Return the top k document ids and scores using QLM (Query Likelihood Model).

    Arguments:
        query: query string
        topk: number of results to return
        data_path: path to processed data
        alpha: document-collection interpolation constant
        normalize: whether to normalize probabilities with log
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Load the processed data
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    words = data['words']
    models = [LanguageModel(**model) for model in data['language_models']]
    models_map = {model.id: model for model in models}
    collection_model = LanguageModel(**data['collection_model'])

    # Clean and tokenize the query
    query = [word for word in clean_words(query) if word in words]
    if not query:
        return []

    # Calculate scores using language models
    scores = {}
    for doc_id, model in models_map.items():
        scores[doc_id] = calculate_interpolated_sentence_probability(
            model, collection_model, query, alpha, normalize)

    # Rank scores and return top k results
    rank = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return rank[:topk]


def test_qlm_search():
    import pandas as pd
    import time
    test_data = pd.read_csv('test_queries.csv')
    total_ms = 0
    for i, expected, query in test_data.itertuples():
        print(f'{i+1:2}. {query=}, {expected=}')
        input_ = (query, 3, 'test_index.json', 0.15, True)
        start = time.time()
        output = qlm_search(*input_)
        end = time.time()
        time_taken = (end - start) * 1000
        for j, (doc_id, score) in enumerate(output):
            doc = get_doc(doc_id, 'test_data.csv')
            print(f'   {j+1:2}) {doc.id=:7}, {score=}, {doc.title=}')
        print(f'    {time_taken=:.0f}ms\n')
        total_ms += time_taken
    avg_time_taken = total_ms / (i + 1)
    latency = 1000 / avg_time_taken
    print(f'in {i+1} queries: {avg_time_taken=:.0f}ms, {latency=:.2f} queries/s')


test_qlm_search()
