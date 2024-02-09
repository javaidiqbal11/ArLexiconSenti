import pandas as pd

from helpers import get_unique_emoji_scores, clean_tweet, lexicons

df = pd.read_csv("data/binary_data.csv")


def score_binary(tweet: str):
    # s, ecount = get_unique_emoji_scores(tweet)
    # word_count = ecount
    # score = s
    score = 0
    word_count = 0
    tweet = clean_tweet(tweet)
    for word in tweet.split():
        if word in lexicons:
            score += lexicons[word]
        word_count += 1
    if word_count == 0:
        return 0
    else:
        threshold = score / word_count
        if threshold >= 0.0:
            return 1
        elif threshold < 0.0:
            return -1


df["label"] = df["Tweet"].apply(lambda x: score_binary(x))
print(df["label"].value_counts())
df.to_csv("data/binary_preprocessed_data.csv", index=False)
