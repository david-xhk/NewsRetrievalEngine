from __future__ import annotations

import pickle
from typing import Literal

import torch

from ranker import calculate_bm25, calculate_interpolated_sentence_probability
from ranknet_lstm import RankNetLSTM
from util import (convert_itos, doc_pipeline, fmt_secs, print_search_results,
                  query_pipeline, timed)


def bm25_search(
    query: str,
    processed_data_path: str,
    topk: int | None = None,
) -> list[tuple[int, float]]:
    """Return the top k document ids and scores using BM25.

    Arguments:
        query: query string
        processed_data_path: path to load processed data
        topk: number of results to return
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Load the processed data
    with open(processed_data_path, 'rb') as fp:
        data = pickle.load(fp)
    vocab = data['vocab']
    docs_map = {}
    for doc in data['docs']:
        convert_itos(doc.title, vocab)
        convert_itos(doc.content, vocab)
        docs_map[doc.id] = doc
    inverted_index = data['inverted_index']

    # Process query
    query = query_pipeline(query, vocab, length=None, to='str')

    # Get the ids of all documents with a word in the query
    doc_ids = set(doc_id for word in query for doc_id in inverted_index[word])

    # Get the respective documents
    hits = [docs_map[doc_id] for doc_id in doc_ids]
    if not hits:
        return []

    # Call the BM25 algorithm and return the top k results
    return calculate_bm25(query, hits)[:topk]


def qlm_search(
    query: str,
    processed_data_path: str,
    topk: int | None = None,
    alpha: float = 0.75,
    normalize: bool = False,
) -> list[tuple[int, float]]:
    """Return the top k document ids and scores using the query likelihood model.

    Arguments:
        query: query string
        processed_data_path: path to load processed data
        topk: number of results to return
        alpha: document-collection interpolation constant
        normalize: if set to true, normalize probabilities with log
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Load the processed data
    with open(processed_data_path, 'rb') as fp:
        data = pickle.load(fp)
    vocab = data['vocab']
    models_map = {model.id: model for model in data['language_models']}
    collection_model = data['collection_model']

    # Process query
    query = query_pipeline(query, vocab, length=None, to='str')

    # Calculate scores using language models
    scores = {}
    for doc_id, model in models_map.items():
        scores[doc_id] = calculate_interpolated_sentence_probability(
            model, collection_model, query, alpha, normalize)

    # Rank scores and return top k results
    rank = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return rank[:topk]


def ranknet_lstm_search(
    query: str,
    processed_data_path: str,
    model_path: str,
    topk: int | None = None,
    query_len: int = 50,
    doc_len: int = 200,
) -> list[tuple[int, float]]:
    """Return the top k document ids and scores using RankNetLSTM.

    Arguments:
        query: query string
        processed_data_path: path to load processed data
        model_path: path to load RankNetLSTM model state
        topk: number of results to return
        query_len: query length to trim/pad to
        doc_len: document length to trim/pad to
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Load the processed data
    with open(processed_data_path, 'rb') as fp:
        data = pickle.load(fp)
    docs = data['docs']
    vocab = data['vocab']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = RankNetLSTM(vocab).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Calculate scores using model
    with torch.no_grad():
        doc = [doc_pipeline(doc, vocab, doc_len) for doc in docs]
        doc = torch.stack(doc).to(device)
        query = query_pipeline(query, vocab, query_len)
        query = torch.stack([query for _ in doc]).to(device)
        output = model(query, doc).detach().flatten().tolist()
        scores = {doc.id: score for doc, score in zip(docs, output)}

    # Rank scores and return top k results
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]


def main(
    query: str,
    processed_data_path: str = 'files/test_data_processed_pickle',
    topk: int = 3,
    type: Literal['bm25', 'qlm', 'ranknet-lstm'] = 'bm25',
    verbose: int = 0,
    raw_data_path: str = 'files/test_data.csv',
    alpha: float = 0.75,
    normalize: bool = False,
    ranknet_lstm_model_path: str | None = 'files/ranknet_lstm.pt',
    query_len: int | None = 50,
    doc_len: int | None = 200,
):
    """Make a search query.

    Arguments:
        query: query string
        processed_data_path: path to load processed data
        topk: number of results to return
        type: search algorithm to use
        verbose: the higher the count, the more info is printed from results
        raw_data_path: path to load raw data (must be .csv or .json format)
        alpha: document-collection interpolation constant for qlm search
        normalize: if set to true, normalize probabilities with log for qlm search
        ranknet_lstm_model_path: path to load model state for ranknet-lstm search
        query_len: query length to trim/pad to
        doc_len: document length to trim/pad to
    """
    search_fn = None
    args = [query, processed_data_path]
    kwargs = {'topk': topk}
    if type == 'bm25':
        search_fn = bm25_search
    elif type == 'qlm':
        search_fn = qlm_search
        kwargs['alpha'] = alpha
        kwargs['normalize'] = normalize
    elif type == 'ranknet-lstm':
        search_fn = ranknet_lstm_search
        args.append(ranknet_lstm_model_path)
        kwargs['query_len'] = query_len
        kwargs['doc_len'] = doc_len
    else:
        raise ValueError(f'invalid type provided: {type}')
    results, time_taken = timed(search_fn, args, kwargs)
    header = {'query': repr(query),
              'search': type,
              'latency': fmt_secs(time_taken)}
    print_search_results(results, raw_data_path, verbose, header)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Make a search query.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('query', help='query string', metavar="QUERY")
    parser.add_argument(
        '-d', '--data', default='files/test_data_processed.pickle',
        help='path to load processed data',
        metavar="PATH", dest='processed_data_path')
    parser.add_argument(
        '-k', '--topk', default=3, type=int,
        help='number of results to return',
        metavar='K', dest='topk')
    parser.add_argument(
        '-t', '--type', default='bm25',
        choices=('bm25', 'qlm', 'ranknet-lstm'),
        help='search algorithm to use',
        metavar='T', dest='type')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='the higher the count, the more info is printed from results',
        dest='verbose')
    parser.add_argument(
        '-r', '--raw', default='files/test_data.csv',
        help='path to load raw data (must be .csv or .json format)',
        metavar="PATH", dest='raw_data_path')
    parser.add_argument(
        '--alpha', default=1.0, type=float,
        help='document-collection interpolation constant for qlm search',
        metavar='A', dest='alpha')
    parser.add_argument(
        '--normalize', action='store_true',
        help='if set to true, normalize probabilities with log for qlm search',
        dest='normalize')
    parser.add_argument(
        '--rnlstm', default='files/test_data.csv',
        help='path to load model state for ranknet-lstm search',
        metavar="PATH", dest='ranknet_lstm_model_path')
    parser.add_argument(
        '--query-len', default=50, type=int,
        help='query length to trim/pad to',
        metavar='QL', dest='query_len')
    parser.add_argument(
        '--doc-len', default=200, type=int,
        help='document length to trim/pad to',
        metavar='DL', dest='doc_len')
    args = parser.parse_args()
    main(**vars(args))
