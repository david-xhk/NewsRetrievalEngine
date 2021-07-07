from typing import Text
import pandas as pd
import matplotlib.pyplot as plt
import os
from langdetect import detect


if os.path.isfile('SUTD_NEMO_news_articles.pkl'):
    df = pd.read_pickle('SUTD_NEMO_news_articles.pkl')
else:
    df = pd.read_excel("SUTD_NEMO_news_articles.xlsx")
    df.to_pickle('SUTD_NEMO_news_articles.pkl')

CORPUS_SIZE = 10_000
filtered = []

# Filtering process
for idx, row in df.iterrows():
  if len(row['content']) < 100: # remove paywall block e.g. "please sign in"
    break
  else:
    if detect(row['content']) == 'en': # check if language is english
      filtered.append(row)

filtered_df = pd.DataFrame(filtered, columns=df.columns)

# Reduce to corupus size
filtered_df = filtered_df.sample(n=CORPUS_SIZE)

# Export to csv
filtered_df.to_csv('filtered_data.csv', index=False)
