from util import clean_words, get_doc
import json

index_path = "test.json"

# Load the stored indexer output
with open(index_path, 'r') as fp:
    [docs, words, wtoi, index] = json.load(fp)


def search(query: str, topk: int = 10) -> list[int]:
    """Return the ids of the top k documents for the query.

    Arguments:
        - query: query string
        - topk: number of results to return
    Returns:
        - list of ranked document ids
    """
    # Tokenize, clean and convert the query
    # Use the inverted index to retrieve the ids all documents which contain a word in the query
    # Get the tokenized documents with the ids
    # Call the BM25 algorithm and return the results


def test_search():
    input_ = ("Assault On Congress Building", 1)
    expected = [325084]
    output = search(*input_)
    assert output == expected, 'expected ' + str(
        expected) + ' from search but got ' + str(output)


test_search()
