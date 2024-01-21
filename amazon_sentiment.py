import boto3
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from helpers import tweet_score, id2label, clean_tweet

df = pd.read_csv("data/preprocessed_data.csv")
client = boto3.client(service_name='comprehend', region_name='us-east-1')
text_list = df["Tweet_cleaned"].values.tolist()


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center',
                 bbox=dict(facecolor='red', alpha =.8))


def get_sentiment(tweet):
    tweet = clean_tweet(tweet)
    response = client.detect_sentiment(Text=tweet, LanguageCode='ar')
    return response['Sentiment']




df["aws_sentiment"] = df['Tweet'].apply(lambda x: get_sentiment(x))
df.to_csv("data/amazon_sentiment.csv")
#
