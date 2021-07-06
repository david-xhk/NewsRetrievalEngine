import pandas as pd
import matplotlib.pyplot as plt
import os

if os.path.isfile('df.pkl'):
  df = pd.read_pickle('df.pkl')
else:
  df = pd.read_excel("SUTD_NEMO_news_articles.xlsx")
  df.to_pickle('df.pkl')

# histogram of sources

#news_counts = df['source'].value_counts()[:8]

#plt.bar(range(len(news_counts)), news_counts.values, align='center')
#plt.xticks(range(len(news_counts)), news_counts.index.values, size='small')
#plt.xlabel('News sources')
#plt.xticks(rotation=90)
#plt.ylabel('Frequency')
#plt.tight_layout()
#plt.savefig('news_freq.png')

