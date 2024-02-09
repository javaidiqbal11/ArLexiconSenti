from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from helpers import id2label

df = pd.read_csv("data/lexicon_sentiment.csv")


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
y_pred = df["lexicon_sentiment"].values.tolist()

print("Lexicon Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
trues = y_true[:2154]
preds = y_pred[:2154]

print(multilabel_confusion_matrix(trues, preds))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(trues, preds, display_labels=["positive", "neutral", "negative"],
                                        cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/multilabel_confusion_matrix.png")
plt.close()

plt.clf()
# historgram
counter = Counter(y_true)
positive = counter[1]
negative = counter[-1]
neutral = counter[0]

plt.figure(figsize=(10, 5))

# making the bar chart on the data
plt.bar(["positive", "neutral", "negative"], [positive, neutral, negative])
# calling the function to add value labels
addlabels(["positive", "neutral", "negative"], [positive, neutral, negative])
# giving title to the plot
plt.title("Dataset")
# giving X and Y labels
plt.xlabel("labels")
plt.ylabel("no. of tweets")
plt.savefig("graphs/multi_histogram.png")
plt.close()
plt.clf()
# roc curve
y_train = df["label"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["lexicon_sentiment"].apply(lambda x: id2label(x)).values.tolist()
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_train)
pred = label_binarizer.transform(y_pred)

class_of_interest = "positive"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/multilabel_{class_id}_roc_curve.png")

plt.clf()

class_of_interest = "negative"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/multilabel_{class_id}_roc_curve.png")

class_of_interest = "neutral"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/multilabel_{class_id}_roc_curve.png")
plt.clf()

####################################################################################################################
####################################################################################################################
####################################################################################################################
# Amazon Sentiment
df = pd.read_csv("data/amazon_sentiment.csv")
df["aws_labels"] = df["aws_sentiment"].apply(lambda x: map_sentiment(x))
df["your_labels"] = df["Annotation"]
y_true = df["your_labels"].values.tolist()
y_pred = df["aws_labels"].values.tolist()

print("Amazon sentiment analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
trues = y_true[:2154]
preds = y_pred[:2154]

print(multilabel_confusion_matrix(trues, preds))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(trues, preds, display_labels=["positive", "neutral", "negative"],
                                        cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/aws_multilabel_confusion_matrix.png")
plt.clf()

# roc curve
y_train = df["label"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["aws_labels"].apply(lambda x: id2label(x)).values.tolist()
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_train)
pred = label_binarizer.transform(y_pred)

class_of_interest = "positive"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/aws_multilabel_{class_id}_roc_curve.png")

plt.clf()

class_of_interest = "negative"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/aws_multilabel_{class_id}_roc_curve.png")

class_of_interest = "neutral"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/aws_multilabel_{class_id}_roc_curve.png")
####################################################################################################################
####################################################################################################################
####################################################################################################################
# logistic_regression

df = pd.read_csv("data/logistic_regression.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("Logistic Regression Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
print(multilabel_confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "neutral", "negative"],
                                        cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/logistic_regression_multilabel_confusion_matrix.png")
plt.clf()

# roc curve
y_train = df["y_true"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["logistic_regression_sentiment"].apply(lambda x: id2label(x)).values.tolist()
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_train)
pred = label_binarizer.transform(y_pred)

class_of_interest = "positive"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/logistic_regression_multilabel_{class_id}_roc_curve.png")

plt.clf()

class_of_interest = "negative"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/logistic_regression_multilabel_{class_id}_roc_curve.png")

class_of_interest = "neutral"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/logistic_regression_multilabel_{class_id}_roc_curve.png")
####################################################################################################################
####################################################################################################################
####################################################################################################################
# Random Forest

df = pd.read_csv("data/random_forest_sentiment.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("Random Forest Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
print(multilabel_confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "neutral", "negative"],
                                        cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/random_forest_multilabel_confusion_matrix.png")
plt.clf()

# roc curve
y_train = df["y_true"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["random_forest_sentiment"].apply(lambda x: id2label(x)).values.tolist()
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_train)
pred = label_binarizer.transform(y_pred)

class_of_interest = "positive"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/random_forest_multilabel_{class_id}_roc_curve.png")

plt.clf()

class_of_interest = "negative"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/random_forest_multilabel_{class_id}_roc_curve.png")

class_of_interest = "neutral"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/random_forest_multilabel_{class_id}_roc_curve.png")


####################################################################################################################
####################################################################################################################
####################################################################################################################
# Decision Tree

df = pd.read_csv("data/decision_tree_sentiment.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("Decision Tree Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
print(multilabel_confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "neutral", "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/decision_tree_multilabel_confusion_matrix.png")
plt.clf()

# roc curve
y_train = df["y_true"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["decision_tree_sentiment"].apply(lambda x: id2label(x)).values.tolist()
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_train)
pred = label_binarizer.transform(y_pred)

class_of_interest = "positive"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/decision_tree_multilabel_{class_id}_roc_curve.png")

plt.clf()

class_of_interest = "negative"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/decision_tree_multilabel_{class_id}_roc_curve.png")

class_of_interest = "neutral"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/decision_tree_multilabel_{class_id}_roc_curve.png")

####################################################################################################################
####################################################################################################################
####################################################################################################################
# SVM

df = pd.read_csv("data/svm_sentiment.csv")
y_true = df["y_true"].values.tolist()
y_pred = df["y_pred"].values.tolist()

print("SVM Sentiment Analysis")
print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
print(multilabel_confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "neutral", "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/svm_multilabel_confusion_matrix.png")
plt.clf()

# roc curve
y_train = df["y_true"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["svm_sentiment"].apply(lambda x: id2label(x)).values.tolist()
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_train)
pred = label_binarizer.transform(y_pred)

class_of_interest = "positive"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (neutral & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/svm_multilabel_{class_id}_roc_curve.png")

plt.clf()

class_of_interest = "negative"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & neutral) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/svm_multilabel_{class_id}_roc_curve.png")

class_of_interest = "neutral"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs (positive & negative) sentiment)")
plt.legend()
# plt.show()
plt.savefig(f"graphs/svm_multilabel_{class_id}_roc_curve.png")
