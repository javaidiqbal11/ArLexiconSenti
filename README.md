### Python
Use python 3.10.10

install packages
```shell
pip install -r requirements.txt
```

### Lexicon
Lexicon taken from here to sentiment

```text
https://github.com/nora-twairesh/AraSenti
```

# convert encoding
```shell
iconv -f windows-1256 -t UTF-8 AraSentiLexiconV1.0 > ar_lexicon.txt
```


### Some helpful material
Emoji sentiment is taken from here to help improve lexicon based twitter sentiment.

```text
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144296
```

use emoji scoring from here
```text
https://kt.ijs.si/data/Emoji_sentiment_ranking/
```


### Sentiment Analysis (Binary Class Positive and Negative)
After processing and cleaning tweet sentiment analysis is performed.
```shell
python sentiment_analysis.py
```

### Multilabel Sentiment Analysis (Positive, Negative and Neutral)
After processing and cleaning tweet sentiment analysis is performed.
```shell
python multilabel_sentiment_analysis.py
```

### AWS Comprehend Multilabel Sentiment Analysis (Positive, Negative and Neutral)
After processing and cleaning tweet sentiment analysis is performed.
```shell
python analysis.py
```

Here are the precision, recall, f1 and accuracy scores. 
[Multilabel Sentiment Analysis Arsent Dictionary]

![image](https://github.com/javaidiqbal11/ArLexiconSenti/assets/30682562/ce166c22-2151-4567-be51-65000388ebf6)


Here are the precision, recall, f1 and accuracy scores.
[Multilabel Sentiment Analysis AWS Comprehend]

![image1](https://github.com/javaidiqbal11/ArLexiconSenti/assets/30682562/c328dc5a-9572-4d20-9d59-57add8d15d9a)

