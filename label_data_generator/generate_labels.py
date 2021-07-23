# Pseudocode for creating labelled dataset:
#    For each document in the dataset:
#        Generate the query from its title
#        Get the relevance score for the query and the document using BM25
#        For each other document in n other random documents:
#            Get the relevance score for the query and the document using BM25
#        Calculate the relevance score at the 90th percentile
#        Documents with relevance score >= 90th percentile are relevant, otherwise not relevant
#        Add the sample (query, relevant_docids, irrelevant_docids) to the dataset

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from searcher import BM25Engine
import numpy as np
import json

df = pd.read_csv("test_data.csv")

# Using porter stemmer. Other alternatives: https://www.geeksforgeeks.org/introduction-to-stemming/
ps = PorterStemmer()
engine = BM25Engine("test_index.json")

def generate_query(title: str) -> str:
  text_tokens = word_tokenize(title)
  # Remove stop words and lower case it
  tokens_without_sw = [ps.stem(word.lower()) for word in text_tokens if not word in stopwords.words()]
  return ' '.join(tokens_without_sw)
  
# Loop through each document, generate a query based on title, pick 4 random others
# Calculate the BM25 between query and each of the 5 documents
# Store in queries dictionary, calculate threshold
# Add to queries_document
queries_document = {} # {"query query query...": {relevant: [234324, 2343]. irrelevant: [23908, 230943]}...}
for index, doc in df.iterrows():
  scores = {}
  # Generate query from title
  query = generate_query(doc["title"])
  scores[doc["ID"]] = engine.calculate_query_doc(doc["ID"], query)

  # Pick n random documents 
  random_doc_df = df.sample(n=4)
  for doc_id in random_doc_df['ID'].to_list():
    scores[doc_id] = engine.calculate_query_doc(doc_id, query)

  # Calculate threshold
  threshold = np.percentile(list(scores.values()), 80)

  # Add to document list
  queries_document[query] = {'relevant': [], 'irrelevant': []}
  for document_id, relevance_score in scores.items():
    if relevance_score > threshold:
      queries_document[query]['relevant'].append(document_id)
    else:
      queries_document[query]['irrelevant'].append(document_id)


with open("labelled_data.json", "w") as fp:
  json.dump(queries_document, fp)