from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from helpers import tweet_score, id2label


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center',
                 bbox=dict(facecolor='red', alpha =.8))


df = pd.read_csv("data/clean_final_data.csv")

print(df.head())

df['my_sentiment'] = df['Tweet'].apply(lambda x: tweet_score(x))
df.to_csv("data/output.csv", index=False)


y_true = df["label"].values.tolist()
y_pred = df["my_sentiment"].values.tolist()

# print(precision_score(y_true, y_pred, average="micro"))
# print(recall_score(y_true, y_pred, average="micro"))
# print(f1_score(y_true, y_pred, average="micro"))
#
#
# print(precision_score(y_true, y_pred, average="weighted"))
# print(recall_score(y_true, y_pred, average="weighted"))
# print(f1_score(y_true, y_pred, average="weighted"))


# print(precision_score(y_true, y_pred, average=None))
# print(recall_score(y_true, y_pred, average=None))
# print(f1_score(y_true, y_pred, average=None))
#
# print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["positive", "negative"], digits=4))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/confusion_matrix.png")
plt.close()

# historgram
counter = Counter(y_true)
positive = counter[1]
negative = counter[-1]

plt.figure(figsize=(10, 5))

# making the bar chart on the data
plt.bar(["positive", "negative"], [positive, negative])
# calling the function to add value labels
addlabels(["positive", "negative"], [positive, negative])
# giving title to the plot
plt.title("Dataset")
# giving X and Y labels
plt.xlabel("labels")
plt.ylabel("no. of tweets")
plt.savefig("graphs/histogram.png")
plt.close()


# roc curve
# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot ROC curve with a smooth, visually appealing line
plt.plot(fpr, tpr, color='darkorange', linewidth=3, label='ROC curve (area = %0.2f)' % roc_auc)

# Diagonal line for random classifier
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Lexicon classifier')

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.2)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(fpr, tpr, color='lightblue', alpha=0.2)
plt.savefig("graphs/roc_curve.png")
plt.close()


# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot PR curve with a visually appealing line
plt.plot(recall, precision, color='forestgreen', linewidth=3, label='PR curve (AP = %0.2f)' % average_precision)

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.2)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(recall, precision, color='lightgreen', alpha=0.2)
plt.savefig("graphs/precision_recall.png")