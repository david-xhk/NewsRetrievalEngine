import json
import time

import numpy as np

from engine import BM25Engine
from util import clean_words, read_docs


def main(
    raw_data_path: str = 'files/test_data.csv',
    processed_data_path: str = 'files/test_data_processed.pickle',
    output_path: str = 'files/test_data_labelled.json',
    k: int = 4,
    q: int = 80,
):
    """Create a pseudo-labelled dataset using BM25.

    Arguments:
        raw_data_path: path to load raw data (must be .csv or .json format)
        processed_data_path: path to load processed data
        output_path: path to save output (must be .json format)
        k: number of positive and negative samples to generate
        q: threshold percentile for relevant/irrelevant scores

    Pseudocode:
        For each document in the dataset: \\
            Generate a query from its title \\
            Get the relevance score for the query and the top k documents using BM25 \\
            Get the relevance score for the query and k other random documents using BM25 \\
            Calculate the relevance score at the q-th percentile \\
            Documents with relevance score below q-th percentile are irrelevant, otherwise relevant \\
            Add (query, relevant_docids, irrelevant_docids) to the dataset
    """
    docs = read_docs(raw_data_path)
    num_docs = len(docs)
    print(f'loaded {raw_data_path} with {num_docs} docs')

    engine = BM25Engine(processed_data_path)
    print(f'loaded {processed_data_path}')

    print('start labelling...')
    start_time = time.time()

    doc_ids_orig = [doc.id for doc in docs]
    dataset = []

    for doc in docs:
        doc_ids = doc_ids_orig.copy()
        scores = []

        # Generate query from title
        query = ' '.join(clean_words(doc.title))

        # Get scores of top k documents using BM25
        positive_samples = engine.rank_topk(query, topk=k)
        for doc_id, _ in positive_samples:
            doc_ids.remove(doc_id)
        scores.extend(positive_samples)

        # Get scores of k other random documents
        doc_ids = np.random.choice(doc_ids, size=k, replace=False).tolist()
        negative_samples = engine.rank(query, doc_ids)
        scores.extend(negative_samples)

        # Calculate threshold
        threshold = np.percentile([score for _, score in scores], q)

        # Generate relevant/irrelevant labels
        relevant, irrelevant = [], []
        for doc_id, score in scores:
            (relevant if score > threshold else irrelevant).append(doc_id)

        # Add datum to dataset
        dataset.append((query, relevant, irrelevant))

    end_time = time.time()
    time_taken = end_time - start_time
    speed = num_docs / time_taken
    time_taken = f'{time_taken // 60:.0f}m{time_taken % 60:.0f}s'
    print(f'labelling completed in {time_taken} ({speed:.1f} docs/s)')

    with open(output_path, 'w') as fp:
        json.dump(dataset, fp)
    print(f'result saved at {output_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a pseudo-labelled dataset using BM25.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--in', default='files/test_data.csv',
        help='path to load input data (must be .csv or .json format)',
        metavar="PATH", dest='raw_data_path')
    parser.add_argument(
        '-d', '--data', default='files/test_data_processed.pickle',
        help='path to load processed data',
        metavar="PATH", dest='processed_data_path')
    parser.add_argument(
        '-o', '--out', default='files/test_data_labelled.json',
        help='path to save output (must be .json format)',
        metavar="PATH", dest='output_path')
    parser.add_argument(
        '-k', '--samples', default=5, type=int,
        help='number of positive and negative samples to generate',
        metavar='K', dest='k')
    parser.add_argument(
        '-q', '--threshold', default=80, type=int,
        help='threshold percentile for relevant/irrelevant scores',
        metavar='Q', dest='q')

    args = parser.parse_args()
    main(**vars(args))
