import pandas as pd
import matplotlib.pyplot as plt
import os
from random import randint

if os.path.isfile('SUTD_NEMO_news_articles.pkl'):
    df = pd.read_pickle('SUTD_NEMO_news_articles.pkl')
else:
    df = pd.read_excel("SUTD_NEMO_news_articles.xlsx")
    df.to_pickle('SUTD_NEMO_news_articles.pkl')

EVALUATION_SIZE = 100

if os.path.isfile('labelled.csv'):
  output = pd.read_csv('labelled.csv').to_dict()
else: 
  output = pd.DataFrame(columns=['ID','query'])

random_labels = [randint(0, len(df)) for x in range(EVALUATION_SIZE)]

for label in random_labels:
  print(df.iloc[label]['title'])
  print(df.iloc[label]['content'])
  print()
  query = input()
  output.append({'ID': df.iloc[label]['ID'], 'query': query}, ignore_index=True)
  output.to_csv('labelled.csv')



  
