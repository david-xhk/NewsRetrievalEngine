from __future__ import annotations

import functools
import os
import pickle
from typing import Literal

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers import util as sbert_util

from ranker import calculate_bm25, calculate_interpolated_sentence_probability
from ranknet_lstm import RankNetLSTM
from util import (convert_itos, doc_pipeline, fmt_secs, load_processed_data,
                  print_search_results, query_pipeline, read_docs, timed,
                  tokenized_doc_pipeline)


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
    data = load_processed_data(processed_data_path, convert_to_string=True)
    vocab = data['vocab']
    docs_map = {doc.id: doc for doc in data['docs']}
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
    data = load_processed_data(processed_data_path)
    vocab = data['vocab']
    models_map = {model.id: model for model in data['language_models']}
    collection_model = data['collection_model']

    # Process query
    query = query_pipeline(query, vocab, length=None, to='str')

    # Calculate scores using language models
    scores = []
    for doc_id, model in models_map.items():
        score = calculate_interpolated_sentence_probability(
            model, collection_model, query, alpha, normalize)
        scores.append((doc_id, score))

    # Rank scores and return top k results
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:topk]


@functools.lru_cache(maxsize=1)
def init_ranknet_lstm_search(
    processed_data_path: str = 'files/test_data_processed.pickle',
    model_path: str = 'files/ranknet_lstm.pt',
    doc_len: int = 200,
):
    # Load the processed data
    data = load_processed_data(processed_data_path)
    docs = data['docs']
    vocab = data['vocab']

    # Load the model
    model = RankNetLSTM(vocab)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Calculate scores using model
    with torch.no_grad():
        doc_pipeline = functools.partial(tokenized_doc_pipeline,
                                         vocab=vocab, length=doc_len)
        corpus_embeddings = list(map(doc_pipeline, docs))
        corpus_embeddings = torch.stack(corpus_embeddings)

    if torch.cuda.is_available():
        model = model.to('cuda')
        corpus_embeddings = corpus_embeddings.to('cuda')

    return docs, vocab, model, corpus_embeddings


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
    # Initialize search
    docs, vocab, model, corpus_embeddings = init_ranknet_lstm_search(
        processed_data_path, model_path, doc_len)

    # Process query
    query_embedding = query_pipeline(query, vocab, query_len)
    query_embeddings = torch.stack([query_embedding] * len(corpus_embeddings))
    if torch.cuda.is_available():
        query_embeddings = query_embeddings.to('cuda')

    # Calculate scores using model
    with torch.no_grad():
        scores = model(query_embeddings, corpus_embeddings)
        scores = scores.detach().flatten().tolist()
        scores = [(doc.id, score) for doc, score in zip(docs, scores)]

    # Rank scores and return top k results
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:topk]


@functools.lru_cache(maxsize=1)
def init_sbert_search(raw_data_path: str = 'files/test_data.csv'):
    docs = read_docs(raw_data_path)

    bi_encoder_name = 'msmarco-MiniLM-L-6-v3'
    bi_encoder = SentenceTransformer(bi_encoder_name)
    cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    cross_encoder = CrossEncoder(cross_encoder_name)

    data_dir, data_filename = os.path.split(raw_data_path)
    data_filename, _ = os.path.splitext(data_filename)
    embedding_filename = f'{data_filename}-MiniLM_embeddings.pt'
    corpus_embeddings_path = os.path.join(data_dir, embedding_filename)

    if not os.path.exists(corpus_embeddings_path):
        passages = list(map(doc_pipeline, docs))
        corpus_embeddings = bi_encoder.encode(
            passages, convert_to_tensor=True, show_progress_bar=True)
        torch.save(corpus_embeddings, corpus_embeddings_path)
    else:
        corpus_embeddings = torch.load(corpus_embeddings_path)

    if torch.cuda.is_available():
        bi_encoder = bi_encoder.to('cuda')
        corpus_embeddings = corpus_embeddings.to('cuda')

    return docs, corpus_embeddings, bi_encoder, cross_encoder


def sbert_search(
    query: str,
    raw_data_path: str = 'files/test_data.csv',
    topk: int | None = None,
):
    """Return the top k document ids and scores using sbert.

    Reference: https://www.sbert.net/examples/applications/retrieve_rerank/README.html

    Arguments:
        query: query string
        raw_data_path: path to load raw data
        topk: number of results to return
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Initialize search
    docs, corpus_embeddings, bi_encoder, cross_encoder = init_sbert_search(
        raw_data_path)

    # Process query using bi-encoder
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    if torch.cuda.is_available():
        query_embedding = query_embedding.to('cuda')

    # Calculate scores
    top_k = max(20, topk)
    hits = sbert_util.semantic_search(
        query_embedding, corpus_embeddings, top_k=top_k)
    hits = [hit['corpus_id'] for hit in hits[0]]

    # Rerank with cross-encoder
    cross_input = [[query, doc_pipeline(docs[i])] for i in hits]
    scores = cross_encoder.predict(cross_input)
    scores = scores.flatten().tolist()
    scores = [(docs[i].id, score) for i, score in zip(hits, scores)]

    # Rank scores and return top k results
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:topk]


@functools.lru_cache(maxsize=1)
def init_dpr_search(raw_data_path: str = 'files/test_data.csv'):
    docs = read_docs(raw_data_path)

    data_dir, data_filename = os.path.split(raw_data_path)
    data_filename, _ = os.path.splitext(data_filename)
    embedding_filename = f'{data_filename}-dpr_embeddings.pt'
    corpus_embeddings_path = os.path.join(data_dir, embedding_filename)

    if not os.path.exists(corpus_embeddings_path):
        passage_model_name = 'facebook-dpr-ctx_encoder-single-nq-base'
        passage_encoder = SentenceTransformer(passage_model_name)
        if torch.cuda.is_available():
            passage_encoder = passage_encoder.to('cuda')
        passages = list(map(doc_pipeline, docs))
        corpus_embeddings = passage_encoder.encode(
            passages, convert_to_tensor=True, show_progress_bar=True)
        torch.save(corpus_embeddings, corpus_embeddings_path)
    else:
        corpus_embeddings = torch.load(corpus_embeddings_path)

    query_model_name = 'facebook-dpr-question_encoder-single-nq-base'
    query_encoder = SentenceTransformer(query_model_name)

    if torch.cuda.is_available():
        query_encoder = query_encoder.to('cuda')
        corpus_embeddings = corpus_embeddings.to('cuda')

    return docs, corpus_embeddings, query_encoder


def dpr_search(
    query: str,
    raw_data_path: str = 'files/test_data.csv',
    topk: int | None = None,
):
    """Return the top k document ids and scores using DPR.

    Reference: https://www.sbert.net/docs/pretrained-models/dpr.html

    Arguments:
        query: query string
        raw_data_path: path to load raw data
        topk: number of results to return
    Returns:
        ranked list of document id-score tuples (best score first)
    """
    # Initialize search
    docs, corpus_embeddings, query_encoder = init_dpr_search(raw_data_path)

    # Process query
    query_embedding = query_encoder.encode(query, convert_to_tensor=True)
    if torch.cuda.is_available():
        query_embedding = query_embedding.to('cuda')

    # Calculate scores using dot product
    scores = sbert_util.dot_score(query_embedding, corpus_embeddings)
    scores = scores.flatten().tolist()
    scores = [(doc.id, score) for doc, score in zip(docs, scores)]

    # Rank scores and return top k results
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:topk]


def main(
    query: str,
    type: Literal['bm25', 'qlm', 'ranknet-lstm'] = 'bm25',
    topk: int = 3,
    verbose: int = 0,
    interactive: bool = False,
    raw_data_path: str = 'files/test_data.csv',
    processed_data_path: str = 'files/test_data_processed_pickle',
    alpha: float = 0.75,
    normalize: bool = False,
    ranknet_lstm_model_path: str | None = 'files/ranknet_lstm.pt',
    query_len: int | None = 50,
    doc_len: int | None = 200,
):
    """Make a search query.

    Arguments:
        query: query string
        type: search algorithm to use
        topk: number of results to return
        verbose: the higher the count, the more info is printed from results
        interactive: if set to true, start interactive search mode
        raw_data_path: path to load raw data (must be .csv or .json format)
        processed_data_path: path to load processed data
        alpha: document-collection interpolation constant for qlm search
        normalize: if set to true, normalize probabilities with log for qlm search
        ranknet_lstm_model_path: path to load model state for ranknet-lstm search
        query_len: query length to trim/pad to
        doc_len: document length to trim/pad to
    """
    search_fn = None
    args = [None]
    kwargs = {'topk': topk}
    if type == 'bm25':
        search_fn = bm25_search
        args.append(processed_data_path)
    elif type == 'qlm':
        search_fn = qlm_search
        args.append(processed_data_path)
        kwargs['alpha'] = alpha
        kwargs['normalize'] = normalize
    elif type == 'ranknet-lstm':
        search_fn = ranknet_lstm_search
        args.append(processed_data_path)
        args.append(ranknet_lstm_model_path)
        kwargs['query_len'] = query_len
        kwargs['doc_len'] = doc_len
    elif type == 'sbert':
        search_fn = sbert_search
        args.append(raw_data_path)
    elif type == 'dpr':
        search_fn = dpr_search
        args.append(raw_data_path)
    else:
        raise ValueError(f'invalid type provided: {type}')
    while True:
        if query:
            args[0] = query
            results, time_taken = timed(search_fn, args, kwargs)
            header = {'query': repr(query),
                      'search': type,
                      'latency': fmt_secs(time_taken)}
            print_search_results(results, raw_data_path, verbose, header)
        if interactive:
            query = input('Next query? (Press Ctrl+C to exit)>')
        else:
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Make a search query.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('query', help='query string', metavar="QUERY")
    parser.add_argument(
        '-t', '--type', default='bm25',
        choices=('bm25', 'qlm', 'ranknet-lstm', 'sbert', 'dpr'),
        help='search algorithm to use',
        metavar='T', dest='type')
    parser.add_argument(
        '-k', '--topk', default=3, type=int,
        help='number of results to return',
        metavar='K', dest='topk')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='the higher the count, the more info is printed from results',
        dest='verbose')
    parser.add_argument(
        '-i', '--interactive', action='store_true',
        help='if set to true, start interactive search mode',
        dest='interactive')
    parser.add_argument(
        '-r', '--raw', default='files/test_data.csv',
        help='path to load raw data (must be .csv or .json format)',
        metavar="PATH", dest='raw_data_path')
    parser.add_argument(
        '-d', '--data', default='files/test_data_processed.pickle',
        help='path to load processed data',
        metavar="PATH", dest='processed_data_path')
    parser.add_argument(
        '--alpha', default=1.0, type=float,
        help='document-collection interpolation constant for qlm search',
        metavar='A', dest='alpha')
    parser.add_argument(
        '--normalize', action='store_true',
        help='if set to true, normalize probabilities with log for qlm search',
        dest='normalize')
    parser.add_argument(
        '--rnlstm', default='files/ranknet_lstm.pt',
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
