from doctypes import Document
from util import clean_words, get_doc
from ranker import calculate_bm25
import nltk
import json


def search(query: str, topk: int = 10, index_path: str = "test.json") -> list[int]:
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

    # Use the inverted index to retrieve the ids of all documents that contain a word in the query
    docs_ = list(set(docid for word in query for docid in index[word]))

    # Get the tokenized documents with the ids
    for i, docid in enumerate(docs_):
        docs_[i] = docs_map[docid]
    
    query = [wtoi[word] for word in query]

    # Call the BM25 algorithm and return the results
    results = calculate_bm25(query, docs_, topk)

    if results:
        results = [tup[0] for tup in results]
    
    return results


def test_search():
    input_ = ("Assault On Congress Building", 3)
    expected = [325084]
    output = search(*input_)
    for docid in output:
        print(get_doc(docid, "test_data.csv", "csv"), "\n")
    assert output == expected, 'expected ' + str(
        expected) + ' from search but got ' + str(output)


test_search()
