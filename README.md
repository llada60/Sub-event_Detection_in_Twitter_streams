# Sub-event Detection in Twitter Streams - NLP Kaggle Competitition

[//]: # (The badges)
[![stars](https://img.shields.io/github/stars/llada60/Sub-event_Detection_in_Twitter_streams?style=social)]()
[![license](https://img.shields.io/github/license/llada60/Sub-event_Detection_in_Twitter_streams)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Sub--event%20Detection%20in%20Twitter%20Streams-blue?logo=kaggle)](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams)

---
CSC_51054_EP Machine and Deep Learning, Fall 2024

Team LLY Members: **_Ziyi LIU, Ling LIU, Yixing YANG_**

---

This repository contains the code for the Kaggle
competition: [Sub-event Detection in Twitter streams](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams), a binary classification task.

Our team, "LLY", achieved **_1st place on the private leaderboard and 12th place on the public leaderboard_**.

## Project Goal

The goal of this project is to detect sub-events within Twitter streams related to specific main events. This involves
processing tweets to identify and classify event-related information.

## File Structure

```
.
├── data/                     # Contains raw and processed datasets (ignored by .gitignore)
│   └── challenge_data/
├── src/
│   ├── data.py               # Advanced data preprocessing and embedding generation
│   └── model.py              # RNN model implementation
├── challenge_data.py         # Initial script for data loading and basic preprocessing
├── cnn.ipynb                 # Jupyter notebook for CNN model experimentation
├── lstm.ipynb                # Jupyter notebook for LSTM model experimentation
├── rnn.ipynb                 # Jupyter notebook for RNN model experimentation
├── machine_learnings.ipynb   # Jupyter notebook for various traditional ML models
├── Logistic.ipynb            # Jupyter notebook for Logistic Regression model
├── voting.ipynb              # Jupyter notebook for combining models using a voting classifier
├── requirements.txt          # Python dependencies
├── DataChallengeReport.pdf   # Project report
├── INF554-Challenge-2024.pdf # Competition details
└── README.md                 # This file
```

## Core Logic

### Data Preprocessing

The primary data preprocessing is handled by `src/data.py`, with an initial version in `challenge_data.py`. Key steps
include:

* **Text Cleaning:** Lowercasing, removing URLs, user mentions (while keeping hashtags), punctuation, and numbers.
* **Language Handling:** Translation of non-English tweets to English and emoji conversion to text.
* **Normalization:** Unicode normalisation and contraction expansion.
* **Tokenization, Stopword Removal, and Lemmatization:** Standard NLP techniques to prepare text for modelling.
* **Parallel Processing:** Utilized for efficient preprocessing of large datasets.
* **Embedding Generation:** `src/data.py` uses pre-trained GloVe embeddings (glove-twitter-200) to convert tweets into
  numerical vectors.

### Models

Various models were explored and implemented:

* **RNN:** A custom `RNNBinaryClassifier` is defined in `src/model.py` using PyTorch.
* **CNN, LSTM:** Explored in their respective Jupyter notebooks (`cnn.ipynb`, `lstm.ipynb`).
* **Traditional Machine Learning Models:** Experiments with models like Logistic Regression are found in
  `machine_learnings.ipynb` and `Logistic.ipynb`.
* **Voting Classifier:** A `voting.ipynb` notebook details the combination of different models to improve performance.

## Setup and Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/llada60/Sub-event_Detection_in_Twitter_streams.git
   cd Sub-event_Detection_in_Twitter_streams
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Data:** The `data/challenge_data/` directory is excluded by `.gitignore`. You will need to download the
   competition data from Kaggle and place it in this directory.
4. **Run Preprocessing:**
   The `src/data.py` script can be run to perform the full preprocessing pipeline. It saves intermediate and final
   processed files.
5. **Explore Notebooks:** The Jupyter notebooks (`*.ipynb`) contain the model training, experimentation, and evaluation
   logic. Open and run these using Jupyter Lab or Jupyter Notebook.

## Key Libraries

* [PyTorch](https://pytorch.org/)
* [Gensim](https://radimrehurek.com/gensim/)
* [NLTK](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [Tqdm](https://tqdm.github.io/)
* [Emoji](https://pypi.org/project/emoji/)
* [Langdetect](https://pypi.org/project/langdetect/)
* [Googletrans](https://pypi.org/project/googletrans/)

---
