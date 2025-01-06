# Sub-event_Detection_in_Twitter_streams

---

CSC_51054_EP - Machine and Deep Learning, Fall 2024

Team LLY
Members: Ziyi LIU, Ling LIU, Yixing YANG

Competition link: [Sub-event Detection in Twitter streams](https://www.kaggle.com/t/946c29c13d024ffcad34ab0b40e85688)
**Rank 1st in the competition.**

## Introduction

Project Structure
```
sub-event-data/
├── data/
│    └── challenge_data/
│         ├── baseline.py
│         ├── dummy_predictions.csv
│         ├── eval_tweets/
│         │    └── *.csv
│         ├── logistic_predictions.csv
│         └── train_tweets/
│             └── *.csv
├── models/
├── README.md
├── report/
├── requirements.txt
└── sample.ipynb
```

## Dataset Description

The dataset provided includes tweets from World Cup games across the 2010 and 2014 tournaments. It has been split into time periods, where each time period has been annotated with a binary label (0 or 1) indicating whether it contains references to any of the following sub-event types 'full time', 'goal', 'half time', 'kick off', 'other', 'owngoal', 'penalty', 'red card', 'yellow card'.

You are given the following files:

- train_tweets/*.json: This directory contains all annotated tweets split by their corresponding match. Each file contains tweet data divided into time periods, with each entry labeled as 0 or 1 based on the presence of sub-events.
- eval_tweets/*.json: This directory contains all tweets that need to be annotated
- baseline.py: This script contains two simple baselines, a Logistic Regression classifier and a Dummy Classifier that always predicts the most frequent class of the training set. You can use the code provide here as a starting point on how to read and process the data and
- logistic_predictions.csv and dummy_predictions.csv: sample submission files in the correct format for the two provided baseline classifiers.

The dataset includes the following columns:

- **ID:** An identifier which is the combination of the following two IDs
- **MatchID:** An identifier for each football match.
- **PeriodID:** An identifier for the time period within the match. Each period is 1 minute long.
- **EventType:** A binary label indicating the presence (1) or absence (0) of a sub-event in the given time period.
- **Timestamp:** A Unix timestamp indicating when the tweet was posted.
- **Tweet:** The text content of the tweet.

## Task and Evaluation

For each time period in the test set, your model should predict whether a specific sub-event occurred based on the provided tweet data. The evaluation metric for this competition is accuracy. Accuracy measures the proportion of correct predictions your model makes, calculated by dividing the number of correct predictions by the total number of predictions. For this binary classification task, the accuracy metric is defined as follows:

$$ \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i) $$

$$
\begin{array}{rl}
where:& \\
&N: \text{The number of samples in the test set} \\
&\hat{y}_i: \text{The predicted label for the i-th sample} \\
&y_i: \text{The true label for the i-th sample} \\
&\mathbb{1}: \text{The indicator function}
\end{array}
$$

## Provided Source Code

You are given a Python script `(baseline.py)` to help you get started with the challenge.

The script uses `Glove` vectorization with a `Logistic Regression` classifier to make predictions based on the text content of the tweets. A **Dummy Classifier** based on label frequency is also included.

Your task includes extending and building upon this baseline by experimenting with different features and model architectures to improve performance.

## Useful Python Libraries

This section lists some recommended libraries that you may find helpful during the project:

scikit-learn^1: A powerful library for machine learning in Python, useful for implementing and tuning various classification models.

NLTK^2: The Natural Language Toolkit is invaluable for text processing tasks such as tokenization, stemming, and frequency analysis.

spaCy^3: Another NLP library that offers efficient methods for handling large text corpora, including tokenization and named entity recognition.

Gensim^4: Useful for topic modeling and text representation through word embeddings, which can add valuable context to tweet content.

Hugging Face Transformers^5: A popular library providing pretrained NLP models. You may consider fine-tuning models like BERT for this binary classification task.
