import re
import emoji
import pandas as pd


df = pd.read_csv('data/ar_lexicon.txt', sep=',')
emj_scores = pd.read_csv("data/emojis_sentiment.csv")

df.columns = ["word", "score"]
lexicons = {}
for word, score in zip(df.word.values.tolist(), df.score.values.tolist()):
    lexicons[word] = score


def label2id(line: str):
    if line == "positive":
        return 1
    elif line == "negative":
        return -1
    else:
        return 0

def id2label(line: str) -> str:
    if line == 1:
        return "positive"
    elif line == -1:
        return "negative"
    else:
        return "neutral"


def pad_zeros(data, max_len):
    """
    Pads a list of lists with zeros to a maximum length.

    Args:
    data: A list of lists, where each sublist may have a different length.
    max_len: The maximum length to pad the sublists to.

    Returns:
    A new list of lists, where each sublist is padded with zeros to the
    specified maximum length.
    """
    padded_data = []

    for sublist in data:
        padded_sublist = sublist + [0] * (max_len - len(sublist))
        padded_data.append(padded_sublist)
    return padded_data


def get_word_score(word: str) -> float:
    if word in lexicons:
        return lexicons[word]

    return 0.0


def get_emoji_score(emoji: str) -> float:
    row = emj_scores[emj_scores["Char"] == emoji]
    if not row.empty:
        return row["Sentiment score [-1...+1]"].values[0]
    return 0.0


def remove_emoji(line: str):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', line)


def get_unique_emoji_scores(line: str):
    escore = 0
    ecount = 0
    emojis = emoji.distinct_emoji_list(line)
    for em in emojis:
        row = emj_scores[emj_scores["Char"] == em]
        if not row.empty:
            escore += row["Sentiment score [-1...+1]"].values[0]
        ecount += 1
    return escore, ecount


def clean_tweet(tweet):
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#","", temp)
    temp = remove_emoji(temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    # temp = re.sub("[^a-z0-9]"," ", temp)
    # temp = temp.split()
    return temp


def tweet_score(tweet: str):
    s, ecount = get_unique_emoji_scores(tweet)
    word_count = ecount
    score = s
    # score = 0
    # word_count = 0
    tweet = clean_tweet(tweet)
    for word in tweet.split():
        if word in lexicons:
            score += lexicons[word]
        word_count += 1
    if word_count == 0:
        return 0
    else:
        threshold = score / word_count
        if threshold > 0.02:
            return 1
        elif threshold < -0.02:
            return -1
        else:
            return 0


def binary_tweet_score(tweet: str):
    s, ecount = get_unique_emoji_scores(tweet)
    word_count = ecount
    score = s
    # score = 0
    # word_count = 0
    tweet = clean_tweet(tweet)
    for word in tweet.split():
        if word in lexicons:
            score += lexicons[word]
        word_count += 1
    if word_count == 0:
        return 0
    else:
        threshold = score / word_count
        if threshold >= 0.00:
            return 1
        elif threshold < 0.0:
            return -1