from engine import BM25Engine
from util import clean_words, read_docs
import numpy as np
import json
import time


def main(
    docs_path: str = 'test_data.csv',
    data_path: str = 'test_data_processed.json',
    out_path: str = 'test_data_labelled.json',
    k: int = 4,
    q: int = 80,
):
    """Create a pseudo-labelled dataset using BM25.

    Arguments:
        docs_path: path to input file (must be .csv or .json format)
        data_path: path to data file (must be .json format)
        out_path: output path (must be .json format)
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
    docs = read_docs(docs_path)
    num_docs = len(docs)
    print(f'loaded {docs_path} with {num_docs} docs')

    engine = BM25Engine(data_path)
    print(f'loaded {data_path}')

    print('start labelling...')
    start_time = time.time()

    doc_ids_orig = [doc.id for doc in docs]
    dataset = []

    for doc in docs:
        doc_ids = doc_ids_orig.copy()
        scores = []

        # Generate query from title
        query = ' '.join(clean_words(doc.title, do_stem=False))

        # Pick top k documents using BM25
        positive_samples = engine.rank_topk(query, topk=k)
        for doc_id, _ in positive_samples:
            doc_ids.remove(doc_id)
        scores.extend(positive_samples)

        # Pick k other random documents
        doc_ids = np.random.choice(doc_ids, size=k, replace=False).tolist()
        negative_samples = engine.rank(query, doc_ids)
        scores.extend(negative_samples)

        # Calculate threshold
        threshold = np.percentile([score[1] for score in scores], q)

        # Add to dataset
        datum = {
            'query': query,
            'relevant_docids': [],
            'irrelevant_docids': [],
        }
        for doc_id, score in scores:
            label = 'relevant' if score > threshold else 'irrelevant'
            datum[f'{label}_docids'].append(doc_id)
        dataset.append(datum)

    end_time = time.time()
    time_taken = end_time - start_time
    speed = num_docs / time_taken
    time_taken = f'{time_taken // 60:.0f}m{time_taken % 60:.0f}s'
    print(f'labelling completed in {time_taken} ({speed:.1f} docs/s)')

    with open(out_path, 'w') as fp:
        json.dump(dataset, fp)
    print(f'wrote result to {out_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a pseudo-labelled dataset using BM25.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--in', default='test_data.csv',
        help='path to input file (must be .csv or .json format)',
        metavar="PATH", dest='docs_path')
    parser.add_argument(
        '-d', '--data', default='test_data_processed.json',
        help='path to data file (must be .json format)',
        metavar="PATH", dest='data_path')
    parser.add_argument(
        '-o', '--out', default='test_data_labelled.json',
        help='output path (must be .json format)',
        metavar="PATH", dest='out_path')
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
