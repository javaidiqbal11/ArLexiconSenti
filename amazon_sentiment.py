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


def map_sentiment(x):
    if x == "MIXED":
        return -1
    elif x == "NEGATIVE":
        return -1
    elif x == "POSITIVE":
        return 1
    elif x == "NEUTRAL":
        return 0


df["aws_sentiment"] = df['Tweet'].apply(lambda x: get_sentiment(x))
df.to_csv("data/amazon_sentiment.csv")
#
# df["aws_labels"] = df["aws_sentiment"].apply(lambda x: map_sentiment(x))
# # df["aws_sentiment"] = df["aws_sentiment"].str.lower()
# y_true = df["label"].values.tolist()
# y_pred = df["aws_labels"].values.tolist()
#
# print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
# print(multilabel_confusion_matrix(y_true, y_pred))
#
# # Confusion matrix
# ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "neutral", "negative"], cmap=plt.cm.Blues)
# # plt.show()
# plt.savefig("graphs/aws_multilabel_confusion_matrix.png")
# plt.close()
#
# plt.clf()
# # historgram
# counter = Counter(y_true)
# positive = counter[1]
# negative = counter[-1]
# neutral = counter[0]
#
# plt.figure(figsize=(10, 5))
#
# # making the bar chart on the data
# plt.bar(["positive", "neutral", "negative"], [positive, neutral, negative])
# # calling the function to add value labels
# addlabels(["positive", "neutral", "negative"], [positive, neutral, negative])
# # giving title to the plot
# plt.title("Dataset")
# # giving X and Y labels
# plt.xlabel("labels")
# plt.ylabel("no. of tweets")
# plt.savefig("graphs/aws_multi_histogram.png")
# plt.close()
# plt.clf()
# # roc curve
# y_train = df["label"].apply(lambda x: id2label(x)).values.tolist()
# y_pred = df["aws_labels"].apply(lambda x: id2label(x)).values.tolist()
# label_binarizer = LabelBinarizer().fit(y_train)
# y_onehot_test = label_binarizer.transform(y_train)
# pred = label_binarizer.transform(y_pred)
#
# class_of_interest = "positive"
# class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
#
# RocCurveDisplay.from_predictions(
#     y_onehot_test[:, class_id],
#     pred[:, class_id],
#     name=f"{class_of_interest} vs the rest",
#     color="darkorange",
#     plot_chance_level=True,
# )
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
# plt.legend()
# # plt.show()
# plt.savefig(f"graphs/aws_multilabel_{class_id}_roc_curve.png")
#
# plt.clf()
#
#
# class_of_interest = "negative"
# class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
# RocCurveDisplay.from_predictions(
#     y_onehot_test[:, class_id],
#     pred[:, class_id],
#     name=f"{class_of_interest} vs the rest",
#     color="darkorange",
#     plot_chance_level=True,
# )
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
# plt.legend()
# # plt.show()
# plt.savefig(f"graphs/aws_multilabel_{class_id}_roc_curve.png")
#
#
# class_of_interest = "neutral"
# class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
# RocCurveDisplay.from_predictions(
#     y_onehot_test[:, class_id],
#     pred[:, class_id],
#     name=f"{class_of_interest} vs the rest",
#     color="darkorange",
#     plot_chance_level=True,
# )
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
# plt.legend()
# # plt.show()
# plt.savefig(f"graphs/aws_multilabel_{class_id}_roc_curve.png")
