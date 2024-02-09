import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from helpers import get_word_score, pad_zeros


# Lexicon-based sentiment analysis using individual word polarities
def lexicon_sentiment_analysis(text):
    tokens = text.split()
    polarities = [get_word_score(word) for word in tokens]

    return polarities


df = pd.read_csv("data/binary_preprocessed_data.csv")
texts = df["Tweet"].values.tolist()


labels = df["label"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Create feature vectors using lexicon word polarities
X_train_lexicon = [lexicon_sentiment_analysis(text) for text in X_train]
X_test_lexicon = [lexicon_sentiment_analysis(text) for text in X_test]
max_length = max(map(len, X_train_lexicon + X_test_lexicon))
X_train_lexicon = pad_zeros(X_train_lexicon, max_length)
X_test_lexicon = pad_zeros(X_test_lexicon, max_length)

rfc_classifier = RandomForestClassifier(max_leaf_nodes=1000, max_depth=50, min_samples_leaf=1)
rfc_classifier.fit(X_train_lexicon, y_train)

# Make predictions on the testing set
y_pred = rfc_classifier.predict(X_test_lexicon)
df = pd.DataFrame(None)
df["y_pred"] = y_pred
df["y_true"] = y_test
df["random_forest_sentiment"] = y_pred
df.to_csv("data/binary_random_forest_sentiment.csv", index=False)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred.astype(int))
print("Accuracy:", accuracy)
