from math import log

def tf_(doc: list) -> dict:
    """Calculate term frequency for all words in a document.
    
    Argument:
        - doc: list of words in the document
    Returns:
        - dict of words to the number of times they occur in the document
    """
    tf = {}
    for word in doc:
        if word not in tf:
            tf[word] = 0
        tf[word] += 1
    return tf

def test_tf_():
    input_ = ['a', 'b', 'c', 'd']
    expected = {'a':1, 'b':1, 'c':1, 'd':1}
    output = tf_(input_)
    assert output == expected, 'expected ' + str(expected) + ' from tf_ but got ' + str(output) 
test_tf_()

def df_(docs: list) -> dict:
    """Calculate document frequency for all words in all documents.

    Argument:
        - docs: list of documents in the corpus (which are themselves lists of words)
    Returns:
        - dict of words to the number of documents they appear in
    """
    df = {}
    for doc in docs:
        for word in doc:
            if word not in df:
                df[word] = 0
            df[word] += 1
    return df

def test_df_():
    input_ = [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
    expected = {'a':1, 'b':2, 'c':3, 'd':2, 'e':1}
    output = df_(input_)
    assert output == expected, 'expected ' + str(expected) + ' from df_ but got ' + str(output) 
test_df_()

def idf_(df: dict, corpus_size: int) -> dict:
    """Calculate inverse document frequency.

    Arguments:
        - df: dict of words to their document frequencies (output of df_)
        - corpus_size: number of documents in the corpus (len of docs)
    Returns:
        - dict of words to their inverse document frequencies
    """
    idf = {}
    for word, freq in df.items():
        idf[word] = round(log(corpus_size / freq), 2)
    return idf

def test_idf_():
    input_ = {'a':1, 'b':2, 'c':3, 'd':2, 'e':1}, 3
    expected = {'a':1.1, 'b':0.41, 'c':0.0, 'd':0.41, 'e':1.1}
    output = idf_(*input_)
    assert output == expected, 'expected ' + str(expected) + ' from idf_ but got ' + str(output) 
test_idf_()

def bm25_(query: list, doc: list, docs: list, k1: float = 1.2, b: float = 0.75):
    """Calculate the BM25 retrieval status value for a query and document.

    Arguments:
        - query: list of words in the query
        - doc: list of words in the document
        - docs: list of documents which contain a word in the query (which are themselves lists of words)
        - k1: document frequency scaling factor
        - b: document length scaling factor
    Returns:
        - BM25 retrieval status value for the query and document
    """
    rsv_d = 0.0
    N = len(docs)
    L_d = len(doc)
    L_ave = sum(map(len, docs)) / N  # calculate average document length
    tf_d = tf_(doc)
    df = df_(docs)
    idf = idf_(df, N)
    for t in query:
        if t not in tf_d:
            continue
        idf_t = idf[t]
        tf_td = tf_d[t]
        rsv_td = idf_t * (((k1+1)*tf_td) / ((k1*((1-b)+b*(L_d/L_ave))+tf_td)))
        rsv_d += rsv_td
    return round(rsv_d, 2)

def test_bm25_():
    input_ = ['b', 'c', 'e'], ['b', 'c', 'd'], [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
    expected = 0.41
    output = bm25_(*input_)
    assert output == expected, 'expected ' + str(expected) + ' from bm25_ but got ' + str(output) 
test_bm25_()

