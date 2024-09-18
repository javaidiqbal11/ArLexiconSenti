# Arbic Tweets Lexicon, AWS Comprehend and Machine Learning Based Sentiment Analysis 

## Setup
```bash
Install Python 3.10
```

[Install PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)



[Visual Studio Code](https://code.visualstudio.com/download)


## Install Packages
```shell
pip install -r requirements.txt
```

## Lexicon
Lexicon taken from here to sentiment

```text
https://github.com/nora-twairesh/AraSenti
```

## Convert Encoding
```shell
iconv -f windows-1256 -t UTF-8 AraSentiLexiconV1.0 > ar_lexicon.txt
```


## Some Helpful Materials
Emoji sentiment is taken from here to help improve lexicon based twitter sentiment.

```text
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144296
```

use emoji scoring from here
```text
https://kt.ijs.si/data/Emoji_sentiment_ranking/
```


## Sentiment Analysis
After pre-processing and cleaning tweet sentiment analysis is performed.
```shell
python lexicon_sentiment.py
```

Here are the precision, recall, f1 and accuracy scores.
```text
              precision    recall  f1-score

    positive     0.9988    0.9662    0.9822
     neutral     0.8296    0.5304    0.6471
    negative     0.8916    0.9987    0.9421

    accuracy                         0.9379
   macro avg     0.9067    0.8317    0.8571
weighted avg     0.9374    0.9379    0.9331
```

## Amazon Comprehend Sentiment
Sentiment analysis performed using Amazon Comprehend service for Arabic language.
Here are the results.
```text
              precision    recall  f1-score

    positive     0.3171    0.4223    0.3622
     neutral     0.5521    0.7354    0.6307
    negative     0.6602    0.3069    0.4190

    accuracy                         0.5190
   macro avg     0.5098    0.4882    0.4707
weighted avg     0.5542    0.5190    0.5048
```

## Machine Learning
We used logistic regression to perform sentiment analysis. Here are the results.
```text
             precision    recall  f1-score 

    positive     0.9234    0.9865    0.9539
     neutral     0.8182    0.3058    0.4452
    negative     0.9194    0.9780    0.9478

    accuracy                         0.9178
   macro avg     0.8870    0.7568    0.7823
weighted avg     0.9116    0.9178    0.9027
```

Another model used to classify tweets is random forest. Here are the results
```text
              precision    recall  f1-score

    positive     0.8453    0.9486    0.8940
     neutral     0.7500    0.1000    0.1765
    negative     0.8607    0.9135    0.8863

    accuracy                         0.8510
   macro avg     0.8187    0.6540    0.6522
weighted avg     0.8425    0.8510    0.8208
```
Decision tree result.
```text
Decision Tree Sentiment Analysis
              precision    recall  f1-score

    positive     0.7998    0.8289    0.8141
     neutral     0.1429    0.0449    0.0684
    negative     0.7351    0.8039    0.7680

    accuracy                         0.7539
   macro avg     0.5593    0.5593    0.5502
weighted avg     0.7192    0.7539    0.7337
```
SVM results
```text
SVM Sentiment Analysis
              precision    recall  f1-score

    positive     0.9184    0.9621    0.9398
     neutral     0.6900    0.3333    0.4495
    negative     0.9149    0.9727    0.9429

    accuracy                         0.9062
   macro avg     0.8411    0.7561    0.7774
weighted avg     0.8950    0.9062    0.8940
```

## Binary Sentiment Analysis
Here are score for Binary Sentiment Analysis...
```text
Lexicon Sentiment Analysis
              precision    recall  f1-score

    positive     0.9972    0.9685    0.9826
    negative     0.9678    0.9971    0.9823

    accuracy                         0.9824
   macro avg     0.9825    0.9828    0.9824
weighted avg     0.9829    0.9824    0.9825

Logistic Regression Sentiment Analysis
              precision    recall  f1-score

    positive     0.9599    0.9659    0.9629
    negative     0.9631    0.9566    0.9599

    accuracy                         0.9615
   macro avg     0.9615    0.9613    0.9614
weighted avg     0.9615    0.9615    0.9615

Random Forest Sentiment Analysis
              precision    recall  f1-score

    positive     0.8999    0.9266    0.9131
    negative     0.9244    0.8969    0.9105

    accuracy                         0.9118
   macro avg     0.9122    0.9118    0.9118
weighted avg     0.9122    0.9118    0.9118

Decision Tree Sentiment Analysis
              precision    recall  f1-score

    positive     0.8487    0.8472    0.8479
    negative     0.8351    0.8367    0.8359

    accuracy                         0.8422
   macro avg     0.8419    0.8419    0.8419
weighted avg     0.8422    0.8422    0.8422

SVM Sentiment Analysis
              precision    recall  f1-score

    positive     0.9529    0.9467    0.9498
    negative     0.9478    0.9539    0.9508

    accuracy                         0.9503
   macro avg     0.9504    0.9503    0.9503
weighted avg     0.9503    0.9503    0.9503
```

