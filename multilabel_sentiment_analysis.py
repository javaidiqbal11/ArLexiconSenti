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


y_true = df["label"].values.tolist()
y_pred = df["my_sentiment"].values.tolist()

print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"], digits=4))
print(multilabel_confusion_matrix(y_true, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["positive", "neutral", "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("graphs/multilabel_confusion_matrix.png")
plt.close()


# roc curve
y_train = df["label"].apply(lambda x: id2label(x)).values.tolist()
y_pred = df["my_sentiment"].apply(lambda x: id2label(x)).values.tolist()
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
plt.title("One-vs-Rest ROC curves:\nPositive vs (neutral & negative) sentiment)")
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
