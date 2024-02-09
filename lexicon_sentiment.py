import pandas as pd

from helpers import tweet_score


df = pd.read_csv("data/preprocessed_data.csv")
print(df.head())

df['lexicon_sentiment'] = df['Tweet'].apply(lambda x: tweet_score(x))
df.to_csv("data/lexicon_sentiment.csv", index=False)
