import boto3
import pandas as pd

df = pd.read_csv("data/updated.csv")
client = boto3.client(service_name='comprehend', region_name='us-east-1')
text_list = df["Tweet_cleaned"].values.tolist()


def get_sentiment(tweet):
    response = client.detect_sentiment(Text=tweet, LanguageCode='ar')
    return response['Sentiment']


df["aws_sentiment"] = df['Tweet_cleaned'].apply(lambda x: get_sentiment(x))
df.to_csv("data/amazon_sentiment.csv")
