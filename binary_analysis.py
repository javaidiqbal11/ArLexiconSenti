from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from helpers import id2label

df = pd.read_csv("data/binary_lexicon_sentiment.csv")


def map_sentiment(x):
    if x == "MIXED":
        return -1
    elif x == "NEGATIVE":
        return -1
    elif x == "POSITIVE":
        return 1
    elif x == "NEUTRAL":
        return 0


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center',
                 bbox=dict(facecolor='red', alpha=.8))


y_true = df["label"].values.tolist()
y_pred = df["binary_lexicon_sentiment"].values.tolist()

print("Lexicon Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "negative"], digits=4))


print(confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("binary_graphs/confusion_matrix.png")
plt.close()

plt.clf()

# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot PR curve with a visually appealing line
plt.plot(recall, precision, color='forestgreen', linewidth=4, label='PR curve (AP = %0.2f)' % average_precision)

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(recall, precision, color='lightgreen', alpha=0.2)

plt.savefig("binary_graphs/lexicon_roc_curve.png")
####################################################################################################################
####################################################################################################################
####################################################################################################################
# logistic_regression

df = pd.read_csv("data/binary_logistic_regression.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("Logistic Regression Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive",  "negative"], digits=4))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive",  "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("binary_graphs/logistic_regression_confusion_matrix.png")
plt.clf()


# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot PR curve with a visually appealing line
plt.plot(recall, precision, color='forestgreen', linewidth=4, label='PR curve (AP = %0.2f)' % average_precision)

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(recall, precision, color='lightgreen', alpha=0.2)

plt.savefig("binary_graphs/logistic_regression_roc_curve.png")

####################################################################################################################
####################################################################################################################
####################################################################################################################
# Random Forest

df = pd.read_csv("data/binary_random_forest_sentiment.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("Random Forest Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive",  "negative"], digits=4))
print(confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive",  "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("binary_graphs/random_forest_binary_confusion_matrix.png")
plt.clf()


# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot PR curve with a visually appealing line
plt.plot(recall, precision, color='forestgreen', linewidth=4, label='PR curve (AP = %0.2f)' % average_precision)

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(recall, precision, color='lightgreen', alpha=0.2)

plt.savefig("binary_graphs/random_forest_roc_curve.png")
####################################################################################################################
####################################################################################################################
####################################################################################################################
# Decision Tree

df = pd.read_csv("data/binary_decision_tree_sentiment.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("Decision Tree Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive",  "negative"], digits=4))
print(confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive",  "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("binary_graphs/decision_tree_binary_confusion_matrix.png")
plt.clf()

# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot PR curve with a visually appealing line
plt.plot(recall, precision, color='forestgreen', linewidth=4, label='PR curve (AP = %0.2f)' % average_precision)

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(recall, precision, color='lightgreen', alpha=0.2)

plt.savefig("binary_graphs/decision_tree_roc_curve.png")

####################################################################################################################
####################################################################################################################
####################################################################################################################
# SVM

df = pd.read_csv("data/svm_sentiment.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("SVM Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive",  "negative"], digits=4))
print(confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive",  "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("binary_graphs/svm_confusion_matrix.png")
plt.clf()


# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot PR curve with a visually appealing line
plt.plot(recall, precision, color='forestgreen', linewidth=4, label='PR curve (AP = %0.2f)' % average_precision)

# Customize plot elements for a polished look
plt.xlim([-0.05, 1.05])  # Add margins for visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # Add a subtle grid

# Enhanced visual appeal with gridlines and shaded area
plt.fill_between(recall, precision, color='lightgreen', alpha=0.2)

plt.savefig("binary_graphs/svm_roc_curve.png")
