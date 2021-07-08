from time import time_ns
from doctypes import Document
from ranker import calculate_bm25
from util import clean_words, get_doc
import json


def search(query: str, topk: int, index_path: str) -> list[int]:
    """Return the ids of the top k documents for the query.

    Arguments:
        - query: query string
        - topk: number of results to return
        - index_path: path to index
    Returns:
        - list of ranked document ids
    """
    # Load the stored indexer output
    with open(index_path, 'r') as fp:
        [docs, words, wtoi, index] = json.load(fp)
    docs = [Document.from_dict(doc) for doc in docs]
    docs_map = {doc.id: doc for doc in docs}

    # Tokenize, clean and convert the query
    query = [word for word in clean_words(query) if word in words]
    if not query:
        return []

    # Use the inverted index to retrieve the ids of all documents that contain
    # a word in the query
    docs_ = list(set(docid for word in query for docid in index[word]))

    # Get the tokenized documents with the ids
    for i, docid in enumerate(docs_):
        docs_[i] = docs_map[docid]

    # Convert query to word indices
    query = [wtoi[word] for word in query]

    # Call the BM25 algorithm and return the results
    results = calculate_bm25(query, docs_, topk)

    # Select docid
    if results:
        results = [docid for (docid, _) in results]

    return results


def test_search():
    import pandas as pd
    import time

    total_ms = 0
    n = 0

    for expected, query in pd.read_csv(
            "test_queries.csv").itertuples(index=False):
        print(f"{n+1:2}. {query=}, {expected=}")
        input_ = (query, 3, "test_index.json")
        start = time.time()
        output = search(*input_)
        end = time.time()
        time_taken = (end - start) * 1000

        for i, docid in enumerate(output):
            doc = get_doc(docid, "test_data.csv", "csv")
            print(f"   {i+1:2}) {doc.id=:7}, {doc.title=}")
        print(f"    {time_taken=:.0f}ms\n")
        total_ms += time_taken
        n += 1

    avg_time_taken = total_ms / n
    latency = 1000 / avg_time_taken
    print(f"in {n} queries: {avg_time_taken=:.0f}ms, {latency=:.2f} queries/s")


test_search()
