# Sentiment-Analysis
## Project Overview
This project focuses on building and evaluating sentiment analysis models using a dataset of tweets.
The goal is to classify the sentiment of tweets into 'positive', 'negative', or 'neutral'.

## Dataset
The project uses the train.csv and test.csv files from the provided dataset.

train.csv: Contains tweet text, selected text (highlighting the part of the tweet that conveys the sentiment), and the corresponding sentiment label.
test.csv: Contains tweet text and is used for making predictions.

## Data Preprocessing
The following data preprocessing steps were performed:

- Handling missing values: Missing values in both the training and testing datasets were identified and removed.
- Text Cleaning: The text data was cleaned by removing URLs, special characters, converting text to lowercase, removing stop words, and lemmatizing the words.
  
## Feature Engineering:
TF-IDF features were generated from the cleaned tweet text using TfidfVectorizer.
A feature representing the length of the cleaned text was added to both dataframes.

## Exploratory Data Analysis (EDA)
Initial data analysis included visualizing the distribution of sentiments, time of tweets, age of users, and the top 10 countries by tweet count. The relationship between sentiment and these categorical features was also explored through count plots.

## Model Selection and Training
Several classification models were trained and evaluated for sentiment prediction:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC
- Random Forest
The models were trained on the combined TF-IDF features and text length feature. The performance of each model was evaluated using a classification report, including precision, recall, and F1-score.

## Hyperparameter Tuning
Hyperparameter tuning was performed on the Logistic Regression model, which initially showed the best performance based on macro average F1-score. 
GridSearchCV with StratifiedKFold cross-validation was used to find the optimal hyperparameters (C, penalty, solver).

## Results
The tuned Logistic Regression model, with the best hyperparameters {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}, achieved a macro average F1-score of 0.70 on the test set.
This showed a slight improvement compared to the initial Logistic Regression model (0.69).

