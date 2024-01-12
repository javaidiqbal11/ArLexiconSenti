import pandas as pd
from helpers import tweet_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
print(precision_score(y_true, y_pred, average="weighted"))
print(recall_score(y_true, y_pred, average="weighted"))
print(f1_score(y_true, y_pred, average="weighted"))


print(precision_score(y_true, y_pred, average=None))
print(recall_score(y_true, y_pred, average=None))
print(f1_score(y_true, y_pred, average=None))

print(accuracy_score(y_true, y_pred))
