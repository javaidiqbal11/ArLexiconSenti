import pandas as pd

from helpers import binary_tweet_score


df = pd.read_csv("data/binary_preprocessed_data.csv")
print(df.head())

df['binary_lexicon_sentiment'] = df['Tweet'].apply(lambda x: binary_tweet_score(x))
df.to_csv("data/binary_lexicon_sentiment.csv", index=False)
