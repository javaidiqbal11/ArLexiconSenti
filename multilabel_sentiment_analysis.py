from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from helpers import tweet_score, id2label


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center',
                 bbox=dict(facecolor='red', alpha =.8))


df = pd.read_csv("data/preprocessed_data.csv")

print(df.head())

df['my_sentiment'] = df['Tweet'].apply(lambda x: tweet_score(x))
df.to_csv("data/output.csv", index=False)
