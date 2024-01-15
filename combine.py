from sklearn.model_selection import train_test_split
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from helpers import get_word_score, pad_zeros


# Lexicon-based sentiment analysis using individual word polarities
def lexicon_sentiment_analysis(text):
    tokens = text.split()
    polarities = [get_word_score(word) for word in tokens]

    return polarities


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center',
                 bbox=dict(facecolor='red', alpha=.8))


df = pd.read_csv("data/updated.csv")
texts = df["Tweet_cleaned"].values.tolist()
labels = df["my_labels"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create feature vectors using lexicon word polarities
X_train_lexicon = [lexicon_sentiment_analysis(text) for text in X_train]
X_test_lexicon = [lexicon_sentiment_analysis(text) for text in X_test]
max_length = max(map(len, X_train_lexicon + X_test_lexicon))
X_train_lexicon = pad_zeros(X_train_lexicon, max_length)
X_test_lexicon = pad_zeros(X_test_lexicon, max_length)
# Initialize SVM classifier
svm_classifier = SVC(kernel="linear")

# Train the SVM model
svm_classifier.fit(X_train_lexicon, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_lexicon)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"SVM Model Accuracy with Lexicon Word Polarities: {accuracy}")
print("SVM Model Classification Report:")
print(report)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["positive", "negative"], cmap=plt.cm.Blues)
# plt.show()
plt.savefig("./svm_confusion_matrix.png")
plt.close()

# historgram
counter = Counter(y_train + y_test)
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
plt.savefig("./svm_histogram.png")
plt.close()

# roc curve
# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Create the plot
plt.figure(figsize=(8, 6))  # Larger figure for better visibility

# Plot ROC curve with a smooth, visually appealing line
plt.plot(fpr, tpr, color='darkorange', linewidth=3, label='ROC curve (area = %0.2f)' % roc_auc)

# Diagonal line for random classifier
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='SVM classifier')

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
plt.savefig("./svm_roc_curve.png")
plt.close()

# Replace with your actual model predictions and true labels
# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)

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
plt.savefig("./svm_precision_recall.png")
