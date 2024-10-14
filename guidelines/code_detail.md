Detailed understanding of each file, Focuesd over the primary functions, core operations, and purpose in the overall sentiment analysis project.

---

### 1. **amazon_sentiment.py**
   - **Purpose**: This script interfaces with AWS Comprehend to perform sentiment analysis on Arabic tweets. It uses Amazon’s NLP service to analyze the sentiment and save results.
   - **Core Steps**:
     1. **Data Loading**: Reads tweets from `data/preprocessed_data.csv`.
     2. **AWS Client Initialization**: Establishes a connection with AWS Comprehend using `boto3`.
     3. **Sentiment Detection**: Iterates over tweets, cleaning each using `clean_tweet` and analyzing sentiment through `detect_sentiment`.
     4. **Saving Results**: The sentiment results are stored in `data/amazon_sentiment.csv`.

---

### 2. **analysis.py**
   - **Purpose**: This file performs an extensive analysis of the model predictions compared to ground truth labels, providing evaluation metrics and visualizations.
   - **Core Steps**:
     1. **Load Data**: Reads data from `data/lexicon_sentiment.csv`.
     2. **Mapping Sentiments**: Converts sentiment labels to numerical values for analysis.
     3. **Evaluation Metrics**: Generates classification reports, multilabel confusion matrices, and ROC curves for various classes (positive, neutral, negative).
     4. **Visualization**: Saves confusion matrices and ROC curves in `graphs/` for visual insight into model performance.

---

### 3. **binary_analysis.py**
   - **Purpose**: This script provides analysis for binary sentiment classifications, producing metrics and visualizations for binary class labels.
   - **Core Steps**:
     1. **Data Preparation**: Loads binary classification results from `data/binary_lexicon_sentiment.csv`.
     2. **Metrics Calculation**: Produces a classification report and a confusion matrix for binary labels.
     3. **Precision-Recall Curve**: Generates and saves a precision-recall curve to visualize the performance of binary sentiment classification.

---

### 4. **binary_decision_tree.py**
   - **Purpose**: Implements a binary decision tree classifier trained on lexicon-based sentiment features.
   - **Core Steps**:
     1. **Load and Split Data**: Loads data from `data/binary_preprocessed_data.csv` and splits into train/test sets.
     2. **Feature Engineering**: Calculates sentiment polarities for each word in a tweet using `lexicon_sentiment_analysis`.
     3. **Model Training**: Fits a decision tree classifier on the training set.
     4. **Prediction and Output**: Predicts binary sentiment on test data and saves results to `data/binary_decision_tree_sentiment.csv`.

---

### 5. **binary_logistic_regression_sentiment.py**
   - **Purpose**: Trains and evaluates a binary logistic regression model for sentiment classification.
   - **Core Steps**:
     1. **Data Preparation**: Loads tweets and labels, splits them into training and testing sets.
     2. **Feature Engineering**: Converts tweets into lexicon-based feature vectors.
     3. **Model Training and Evaluation**: Trains logistic regression and evaluates its accuracy.
     4. **Output Results**: Saves predictions to `data/binary_logistic_regression.csv`.

---

### 6. **binary_preprocess.py**
   - **Purpose**: Preprocesses tweets for binary sentiment classification by cleaning and scoring words in the tweet.
   - **Core Steps**:
     1. **Data Cleaning**: Removes unwanted characters and emojis from tweets using `clean_tweet`.
     2. **Binary Scoring**: Calculates average sentiment scores using `score_binary`.
     3. **Labeling**: Assigns binary sentiment labels based on score thresholds and outputs to `data/binary_preprocessed_data.csv`.

---

### 7. **binary_random_forest_classifier.py**
   - **Purpose**: Uses a random forest model for binary sentiment classification on lexicon-based features.
   - **Core Steps**:
     1. **Data Splitting**: Splits data from `data/binary_preprocessed_data.csv` into training and testing sets.
     2. **Lexicon Feature Engineering**: Generates features using `lexicon_sentiment_analysis`.
     3. **Random Forest Training**: Fits a random forest model and makes predictions.
     4. **Evaluation and Output**: Saves results to `data/binary_random_forest_sentiment.csv`.

---

### 8. **binary_svm_sentiment.py**
   - **Purpose**: Trains a binary Support Vector Machine (SVM) model for sentiment classification.
   - **Core Steps**:
     1. **Data Loading and Splitting**: Loads binary sentiment data, splits it into training and test sets.
     2. **Lexicon-Based Feature Generation**: Constructs feature vectors based on word sentiment scores.
     3. **Model Training and Prediction**: Trains SVM model and evaluates on test data.
     4. **Save Results**: Outputs predictions to `data/binary_svm_sentiment.csv`.

---

### 9. **decision_tree_sentiment.py**
   - **Purpose**: Implements a decision tree for multiclass sentiment classification.
   - **Core Steps**:
     1. **Data Splitting**: Splits the cleaned data into training and testing sets.
     2. **Feature Engineering**: Constructs feature vectors based on word-level sentiment scores.
     3. **Training**: Trains a decision tree on lexicon-based features.
     4. **Output and Evaluation**: Saves predictions and outputs accuracy score.

---

### 10. **helpers.py**
   - **Purpose**: Provides various utility functions to support data preprocessing and scoring.
   - **Core Functions**:
     - **clean_tweet**: Cleans and normalizes text, removing hashtags, mentions, and URLs.
     - **get_word_score**: Retrieves sentiment score for a word based on lexicons.
     - **pad_zeros**: Pads sequences to a maximum length, making them compatible with model input requirements.
     - **get_emoji_score**: Retrieves sentiment score for emojis.
     - **label2id and id2label**: Maps sentiment labels to integer IDs and vice versa.

---

### 11. **lexicon_sentiment.py**
   - **Purpose**: Computes tweet sentiment scores using a lexicon-based approach, applying scores based on word polarities.
   - **Core Steps**:
     1. **Data Loading**: Loads preprocessed tweets.
     2. **Sentiment Scoring**: Scores each tweet based on lexicon sentiment values.
     3. **Saving Results**: Saves results to `data/lexicon_sentiment.csv`.

---

### 12. **logistic_regression_sentiment.py**
   - **Purpose**: Implements a logistic regression classifier for multiclass sentiment classification.
   - **Core Steps**:
     1. **Feature Creation**: Uses lexicon-based sentiment polarities as feature vectors.
     2. **Training and Prediction**: Trains logistic regression, predicts on test data, and saves results.

---

### 13. **preprocess.py**
   - **Purpose**: Cleans tweets, removes emojis, and calculates average sentiment scores, labeling them based on score thresholds.
   - **Core Steps**:
     1. **Cleaning**: Standardizes and removes irrelevant text features.
     2. **Scoring**: Scores words based on their polarity in the lexicon.
     3. **Label Assignment**: Assigns sentiment labels and outputs to `data/preprocessed_data.csv`.

---

### 14. **random_forest_classifier.py**
   - **Purpose**: Trains a random forest model on lexicon-based features for multiclass sentiment classification.
   - **Core Steps**:
     1. **Data Splitting**: Splits tweets and labels into train/test sets.
     2. **Feature Extraction**: Creates feature vectors based on lexicon word scores.
     3. **Training and Prediction**: Fits the model, predicts on test data, and saves results.

---

### 15. **svm_sentiment.py**
   - **Purpose**: Implements SVM for multiclass sentiment classification.
   - **Core Steps**:
     1. **Feature Engineering**: Converts tweets into lexicon-based feature vectors.
     2. **Model Training and Prediction**: Trains an SVM model and outputs predictions to `data/svm_sentiment.csv`.

---

### 16. **requirements.txt**
   - **Purpose**: Lists all required dependencies, including libraries for NLP (like `scikit-learn` for classification, `pandas` for data handling, `matplotlib` for visualization, and `emoji` for emoji processing).

---

### Data Files in **`data/`**
   - **arabic_tweets.csv**: Original dataset of Arabic tweets for sentiment analysis.
   - **preprocessed_data.csv**: Processed data with cleaned tweets and initial sentiment labeling.
   - **lexicon_sentiment.csv**: Tweets labeled with lexicon-based sentiment analysis.
   - **binary_preprocessed_data.csv**: Binary version of preprocessed data.
   - **emojis_sentiment.csv**: Contains emojis and their sentiment scores for analysis.

---

### Output Visualization Folder **`graphs/`**
   - **Purpose**: Stores plots generated during the analysis (ROC curves, confusion matrices, etc.), providing visual summaries of model performance. 

This detailed explanation provides a file-by-file breakdown of the project structure, highlighting each file’s role and key operations to help with both understanding and hands-on implementation.
