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

# Filter df to remove entries where content is not available (e.g. "Sign in to access data")
# Filter df to remove non-english

for idx, row in df.iterrows():
  if len(row['content']) < 100:
    break
  else:
    if detect(row['content']) == 'en':
      filtered.append(row)

filtered_df = pd.DataFrame(filtered, columns=df.columns)

# Pick a row from every x rows
hop = int(len(filtered_df) / CORPUS_SIZE)

output = filtered_df.iloc[::hop, :]

output.to_csv('filtered_data.csv')
