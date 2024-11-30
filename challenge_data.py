import multiprocessing as mp
import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Basic preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def preprocess_parallel(texts):
    with mp.Pool(mp.cpu_count()) as pool:
        processed_texts = list(tqdm(pool.imap(preprocess_text, texts), total=len(texts), desc="Preprocessing Tweets"))
    return processed_texts


# Read all training files and concatenate them into one dataframe
def load_data(directory):
    li = []
    for filename in tqdm(os.listdir(directory), desc="Loading Data Under " + directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(directory + filename)
            li.append(df)
    df = pd.concat(li, ignore_index=True)
    return df


def preprocess_data() -> tuple:
    data_path = "data/challenge_data/"
    train_data_folder = os.path.join(data_path, "train_tweets/")

    # Try load data directly
    try:
        train_df = pd.read_csv(data_path + 'preprocessed/train_tweets.csv')
        print("Preprocessed data loaded from existing file.")
        X = train_df['Tweet'].values
        y = train_df['EventType'].values
        return X, y
    except FileNotFoundError:
        pass

    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load training data
    train_df = load_data(train_data_folder)
    # Apply preprocessing
    train_df['Tweet'] = preprocess_parallel(train_df['Tweet'])
    # val_data_folder = os.path.join(data_path, "eval_tweets/")
    # val_df = load_data(val_data_folder)
    # val_df['Tweet'] = preprocess_parallel(val_df['Tweet'])

    # Save preprocessed data
    if not os.path.exists(data_path + 'preprocessed/'):
        os.makedirs(data_path + 'preprocessed/')
    train_df.to_csv(data_path + 'preprocessed/train_tweets.csv', index=False)
    print("Preprocessed data saved to file.")

    # Features and Labels
    X = train_df['Tweet'].values
    y = train_df['EventType'].values

    return X, y
