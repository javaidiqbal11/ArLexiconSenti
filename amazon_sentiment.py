import boto3
import pandas as pd

df = pd.read_csv("data/updated.csv")

comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')

text_list = df["Tweet_cleaned"].values.tolist()

sentiment_batch = comprehend.batch_detect_sentiment(TextList=text_list, LanguageCode='ar')


def parse_sentiment_batch(data):
    df = pd.DataFrame([item['SentimentScore'] for item in data['ResultList']])
    df['Sentiment'] = [item.get('Sentiment') for item in data['ResultList']]
    df['Index'] = [item.get('Index') for item in data['ResultList']]
    df.set_index('Index', inplace=True)

    return df


parsed = parse_sentiment_batch(sentiment_batch).head()
parsed.to_csv("data/amazon_sentiment.csv")
