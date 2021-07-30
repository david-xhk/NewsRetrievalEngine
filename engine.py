import pickle

from ranker import calculate_bm25
from util import clean_words, convert_itos, replace_unknown_words


class BM25Engine():
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        # Load the processed data
        with open(self.data_path, 'rb') as fp:
            data = pickle.load(fp)
        self.vocab = data['vocab']
        self.docs_map = {}
        for doc in data['docs']:
            convert_itos(doc.title, self.vocab)
            convert_itos(doc.content, self.vocab)
            self.docs_map[doc.id] = doc
        self.inverted_index = data['inverted_index']

    def _process_query(self, query: str):
        # Clean and tokenize the query and replace unknown words
        query = replace_unknown_words(clean_words(query), self.vocab)

        # Get the ids of all documents with a word in the query
        doc_ids = set(doc_id for word in query
                      for doc_id in self.inverted_index[word])

        # Get the respective documents
        hits = [self.docs_map[doc_id] for doc_id in doc_ids]

        return query, hits

    def rank_one(self, query: str, doc_id: int) -> float:
        query, hits = self._process_query(query)
        if not any(doc.id == doc_id for doc in hits):
            return 0.0
        results = calculate_bm25(query, hits)
        return next(filter(lambda result: result[0] == doc_id, results))[1]

    def rank(self, query: str, doc_ids: list[int]) -> list[tuple[int, float]]:
        scores = dict.fromkeys(doc_ids, 0.0)
        query, hits = self._process_query(query)
        for doc_id, score in calculate_bm25(query, hits):
            if doc_id in scores:
                scores[doc_id] = score
        return list(scores.items())

    def rank_topk(self, query: str, topk: int) -> list[tuple[int, float]]:
        query, hits = self._process_query(query)
        return calculate_bm25(query, hits)[:topk]
