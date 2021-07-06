# from types import Words, Corpus, Index
import nltk
import string

stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)

def clean_(words):
    words_ = []
    for word in nltk.tokenize.word_tokenize(words):
        word = word.encode('ascii', 'ignore').decode() # removes non-ascii characters
        if word and word not in punctuation and not word.isnumeric():
            word = word.lower()
            word = stemmer.stem(word)
            if word not in stopwords:
                words_.append(word)
    return words_
    
def preprocess_(docs):
    for doc in docs:
        doc.title = clean_(doc.title)
        # doc.content = clean_(doc.content)

def words_(docs):
    """Return a list of unique words across all documents."""
    return sorted(set(word for doc in docs for word in doc.title))

def invert_(docs, words):
    """Return the inverted index for all words in the documents."""
    return dict((word, [doc.id for doc in docs if word in doc.title]) for word in words)

def main(docs):
    docs = preprocess_(docs)
    words = words_(docs)
    index = invert_(docs, words)
    return index

def test_main():
    input = ['breakthrough drug for schizophrenia', 'new schizophrenia drugs', 'new approach for treatment of schizophrenia', 'new hopes for schizophrenia patients']
    expected = {'approach': [2], 'breakthrough': [0], 'drug': [0, 1], 'hope': [3], 'new': [1, 2, 3], 'patient': [3], 'schizophrenia': [0, 1, 2, 3], 'treatment': [2]}
    output = main(input)
    assert output == expected, 'expected ' + str(expected) + ' from main but got ' + str(output) 
test_main()

if __name__ == '__main__':
    import argparse 
    import json
    from util import read_docs
    
    parser = argparse.ArgumentParser(description='Generate an inverted index', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('docs_path', help='path to docs')
    parser.add_argument('out_path', help='output path')
    parser.add_argument('--docs_type', metavar='TYPE', help='filetype of docs', default='excel')
    args = parser.parse_args()
    
    docs = read_docs(args.docs_path, args.docs_type)
    index = main(docs)
    with open(args.out_path, 'w') as fp:
        json.dump(index, fp)