This guide organizes the code into three main sections based on the approach used: Lexicon-based, Machine Learning-based, and Amazon Comprehend. Each section provides a detailed explanation of each file, its purpose, and how it integrates with other components of the project.

---

## Project Structure and Running the Code

### Directory Structure Overview

```
ArLexiconSenti/
│
├── data/                           # Stores datasets and processed data files
├── graphs/                         # Stores output graphs from analysis scripts
├── helpers.py                      # Helper functions for text processing and scoring
├── amazon_sentiment.py             # Amazon Comprehend sentiment analysis script
├── analysis.py                     # General analysis for ML-based approaches
├── binary_analysis.py              # Analysis script specifically for binary sentiment analysis
├── preprocess.py                   # General preprocessing for lexicon and ML-based approaches
├── lexicon_sentiment.py            # Lexicon-based sentiment analysis script
├── machine_learning_models/        # Folder for ML-based sentiment analysis models
│   ├── logistic_regression_sentiment.py
│   ├── random_forest_classifier.py
│   ├── svm_sentiment.py
│   ├── decision_tree_sentiment.py
├── requirements.txt                # Dependencies for the project
└── README.md                       # Instructions for setting up and running the project
```

### Running the Code

1. **Install Dependencies**: First, ensure all required packages are installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data**: Start by running `preprocess.py` to clean the tweets, creating a file like `data/preprocessed_data.csv` which will be used across all approaches.
   ```bash
   python preprocess.py
   ```

3. **Choose an Approach**:
   - **Lexicon-Based**: Run `lexicon_sentiment.py` to apply lexicon-based scoring.
   - **Machine Learning-Based**: Choose one of the ML models in `machine_learning_models/` and run the corresponding file (e.g., `random_forest_classifier.py`).
   - **Amazon Comprehend-Based**: Run `amazon_sentiment.py` to use AWS Comprehend for sentiment analysis.

4. **Analysis**: Use `analysis.py` or `binary_analysis.py` to evaluate the model results, generate metrics, and output visualizations.

---

## Section 1: Lexicon-Based Approach

The lexicon-based approach uses predefined word or emoji sentiment scores to evaluate tweets.

### Relevant Files

1. **`preprocess.py`**
   - **Purpose**: This file cleans and processes raw text data, making it suitable for lexicon-based analysis. It normalizes text, removes special characters, and assigns initial sentiment labels based on a scoring function.
   - **Functions**:
     - `clean_tweet`: Removes emojis and unwanted characters.
     - `score`: Scores each tweet by averaging word polarities from the lexicon, assigning labels based on threshold values.
   - **Output**: Saves processed tweets with labels to `data/preprocessed_data.csv`.
   - **Run**:
     ```bash
     python preprocess.py
     ```

2. **`lexicon_sentiment.py`**
   - **Purpose**: Calculates sentiment scores for each tweet using lexicon-based word polarity scores.
   - **Functions**:
     - `tweet_score`: Computes a sentiment score for each tweet by summing word polarities.
   - **Output**: Generates `data/lexicon_sentiment.csv`, containing tweets with lexicon-based sentiment scores.
   - **Run**:
     ```bash
     python lexicon_sentiment.py
     ```

3. **`helpers.py`**
   - **Purpose**: Provides utility functions to support lexicon-based sentiment scoring.
   - **Functions**:
     - `get_word_score`: Retrieves the sentiment polarity score for a given word from the lexicon.
     - `get_emoji_score`: Retrieves a sentiment score for emojis.
     - `pad_zeros`: Pads shorter feature vectors to a standard length for compatibility with ML models.
   - **Integration**: Used across multiple files for standard functions.

4. **`binary_lexicon_sentiment.py`**
   - **Purpose**: A simplified lexicon-based sentiment analysis approach for binary classification.
   - **Functions**:
     - `binary_tweet_score`: Similar to `tweet_score` but tailored for binary labels (positive/negative).
   - **Output**: Generates `data/binary_lexicon_sentiment.csv`, used in binary classification analysis.
   - **Run**:
     ```bash
     python binary_lexicon_sentiment.py
     ```

---

## Section 2: Machine Learning-Based Approach

The machine learning-based approach trains classifiers on sentiment-labeled tweets, using lexicon scores as features.

### Relevant Files

1. **`logistic_regression_sentiment.py`**
   - **Purpose**: Trains and evaluates a logistic regression model for sentiment classification.
   - **Functions**:
     - `lexicon_sentiment_analysis`: Converts each tweet to a lexicon-based feature vector.
     - `pad_zeros`: Pads sequences for uniform feature length.
   - **Output**: Saves predictions to `data/logistic_regression.csv`.
   - **Run**:
     ```bash
     python machine_learning_models/logistic_regression_sentiment.py
     ```

2. **`random_forest_classifier.py`**
   - **Purpose**: Uses a random forest model to classify tweet sentiment.
   - **Functions**:
     - Trains a random forest classifier on lexicon-based features.
     - Evaluates the model on test data and saves predictions.
   - **Output**: Results saved to `data/random_forest_sentiment.csv`.
   - **Run**:
     ```bash
     python machine_learning_models/random_forest_classifier.py
     ```

3. **`svm_sentiment.py`**
   - **Purpose**: Implements Support Vector Machine (SVM) for multiclass sentiment classification.
   - **Functions**:
     - Feature extraction through lexicon-based word scores.
   - **Output**: Saves predictions to `data/svm_sentiment.csv`.
   - **Run**:
     ```bash
     python machine_learning_models/svm_sentiment.py
     ```

4. **`decision_tree_sentiment.py`**
   - **Purpose**: Classifies tweet sentiment using a decision tree model.
   - **Functions**:
     - Generates lexicon-based feature vectors and trains the decision tree.
   - **Output**: Saves predictions in `data/decision_tree_sentiment.csv`.
   - **Run**:
     ```bash
     python machine_learning_models/decision_tree_sentiment.py
     ```

5. **`binary_*_sentiment.py` (e.g., `binary_logistic_regression_sentiment.py`, `binary_random_forest_classifier.py`)**
   - **Purpose**: Binary classifiers for sentiment using various ML models (logistic regression, decision tree, random forest, SVM).
   - **Key Steps**:
     - Train-test split and lexicon feature generation.
     - Binary sentiment classification using the specified model.
     - **Output**: Predictions saved to the corresponding binary sentiment CSV file, e.g., `data/binary_random_forest_sentiment.csv`.

---

## Section 3: Amazon Comprehend Approach

This approach leverages Amazon Comprehend, an NLP service by AWS, for sentiment analysis.

### Relevant Files

1. **`amazon_sentiment.py`**
   - **Purpose**: Performs sentiment analysis on tweets using Amazon Comprehend.
   - **Functions**:
     - `get_sentiment`: Sends each tweet to AWS Comprehend for analysis and retrieves sentiment labels.
   - **AWS Requirements**: Ensure `boto3` is configured with AWS credentials to access Comprehend.
   - **Output**: Saves sentiment-labeled data in `data/amazon_sentiment.csv`.
   - **Run**:
     ```bash
     python amazon_sentiment.py
     ```

---

## Analysis and Visualization

After generating sentiment predictions, these scripts provide evaluations and visualizations.

### Relevant Files

1. **`analysis.py`**
   - **Purpose**: Analyzes the performance of multiclass ML-based sentiment models.
   - **Functions**:
     - Generates classification reports, confusion matrices, and ROC curves for each class.
   - **Output**: Saves analysis plots to `graphs/`.

2. **`binary_analysis.py`**
   - **Purpose**: Evaluates binary sentiment classification results.
   - **Functions**:
     - Generates binary confusion matrices, precision-recall curves, and ROC curves.
   - **Output**: Saves binary analysis visualizations to `binary_graphs/`.

---

### Running an Analysis

After running a classification script, use `analysis.py` for multiclass results or `binary_analysis.py` for binary results to generate visualizations:

```bash
# For multiclass analysis
python analysis.py

# For binary analysis
python binary_analysis.py
```

---

This guide should provide a clear roadmap of the structure, purpose, and execution of each file within the project, ensuring comprehensive understanding and usability.
