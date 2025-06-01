#!/usr/bin/env python3

import multiprocessing as mp
import os
import re
import unicodedata

import emoji
import gensim.downloader as api
import googletrans
import nltk
import numpy as np
import pandas as pd
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.rich import tqdm

# Read all training files and concatenate them into one dataframe
data_root = "../data/challenge_data/"
train_data_path = os.path.join(data_root, "train_tweets/")
eval_data_path = os.path.join(data_root, "eval_tweets/")
save_train_path = os.path.join(data_root, "merged_train.csv")
save_eval_path = os.path.join(data_root, "merged_eval.csv")
save_clean_train_path = os.path.join(data_root, "merged_train_clean.csv")
save_clean_eval_path = os.path.join(data_root, "merged_eval_clean.csv")
model_cache_path = os.path.join(data_root, "model_cache")

# Download resources if not already done
nltk.download('stopwords')
nltk.download('wordnet')
translator = googletrans.Translator()
print(f"Downloaded resources")


def preprocess_text(text):
    """
    emoji2text -> Translate to English -> Unicode Normalization -> Remove URLs -> Remove User Mentions -> Keep Hashtags
    -> Remove Retweet/Reply Indicators -> Lowercasing -> Expand Contractions -> Remove Punctuation -> Remove Numbers
    -> Replace Multiple Spaces -> Tokenization -> Remove Stopwords -> Lemmatization -> Final Join
    """
    # Convert emojis to descriptive text
    text = emoji.demojize(text)
    text = text.replace(":", " ").replace("_", " ")

    # Detect language and translate to English if not already in English
    try:
        lang = detect(text)
        if lang == 'en':
            raise Exception("Already in English")
        # tqdm.write(f"Trans from {lang} to en: {text}")
        # googletrans handles emoji, but you can demojize first if you prefer
        text = translator.translate(text, dest='en').text
    except Exception as e:
        pass

    # Normalize unicode (NFKD) and remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.strip()

    # Remove URLs, User Mentions, Hashtags, Retweet/Reply Indicators
    text = re.sub(r"http\:\S*\s*", " ", text)
    text = re.sub(r"https\:\S*\s*", " ", text)
    text = re.sub(r"www\.\S*\s*", " ", text)
    text = re.sub(r"@(\w+)", r"\1", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"^\"*(RT|rt)(\s+)", " ", text)

    # Lowercasing
    text = text.lower()

    # Expand contractions
    contractions_dict = {
        "you're": "you are", "i've": "i have", "you've": "you have", "you'll": "you will", "i'll": "i will",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
        "haven't": "have not", "hadn't": "had not", "wouldn't": "would not", "doesn't": "does not", "didn't": "did not",
        "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
        "would've": "would have", "should've": "should have", "could've": "could have", "might've": "might have",
        "must've": "must have", "we're": "we are", "they're": "they are", "i'd": "i would", "he'd": "he would",
        "she'd": "she would", "we'd": "we would", "they'd": "they would", "he's": "he is", "she's": "she is",
        "there's": "there is", "that's": "that is", "what's": "what is", "where's": "where is", "who's": "who is",
        "how's": "how is", "let's": "let us", "who're": "who are", "don't": "do not"
    }

    def expand_contractions(s):
        pattern = re.compile(r'({})'.format('|'.join(contractions_dict.keys())))
        return pattern.sub(lambda m: contractions_dict[m.group(0)], s)

    text = expand_contractions(text)

    # Remove punctuation except in special tokens and word chars
    text = re.sub(r"[^\w\s\<\>\:]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)


def preprocess_data() -> (pd.DataFrame, pd.DataFrame):
    # Load the data
    if os.path.exists(save_train_path) and os.path.exists(save_eval_path):
        df_train = pd.read_csv(save_train_path)
        df_eval = pd.read_csv(save_eval_path)
    else:
        li_train = []
        for filename in tqdm(os.listdir(train_data_path), desc="Loading Data (Train)"):
            li_train.append(pd.read_csv(train_data_path + filename))
        df_train = pd.concat(li_train, ignore_index=True)

        li_eval = []
        for filename in tqdm(os.listdir(eval_data_path), desc="Loading Data (Eval)"):
            li_eval.append(pd.read_csv(eval_data_path + filename))
        df_eval = pd.concat(li_eval, ignore_index=True)

        df_train.to_csv(save_train_path, index=False)
        df_eval.to_csv(save_eval_path, index=False)

    print(f"Train data length concat: {len(df_train)}")
    print(f"Eval data length concat: {len(df_eval)}")

    # Apply preprocessing to each tweet
    if os.path.exists(save_clean_train_path) and os.path.exists(save_clean_eval_path):
        df_train = pd.read_csv(save_clean_train_path)
        df_eval = pd.read_csv(save_clean_eval_path)
    else:
        with mp.Pool(mp.cpu_count() * 2) as pool:
            processed_texts = list(
                tqdm(pool.imap(preprocess_text, df_train['Tweet']), total=len(df_train),
                     desc="Preprocessing train tweets"))
        df_train['Tweet'] = processed_texts
        df_train.to_csv(save_clean_train_path, index=False)
        with mp.Pool(mp.cpu_count() * 2) as pool:
            processed_texts = list(
                tqdm(pool.imap(preprocess_text, df_eval['Tweet']), total=len(df_eval),
                     desc="Preprocessing eval tweets"))
        df_eval['Tweet'] = processed_texts
        df_eval.to_csv(save_clean_eval_path, index=False)

    print(f"Train data length clean: {len(df_train)}")
    print(f"Eval data length clean: {len(df_eval)}")

    return df_train, df_eval


def embed_data(df_train: pd.DataFrame, df_eval: pd.DataFrame):
    # Load GloVe model with Gensim's API
    embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
    print("Embedding model loaded.")

    # Function to compute the average word vector for a tweet

    def get_avg_embedding(tweet, model, vector_size=200):
        # Check if the tweet is empty or non-string
        if not isinstance(tweet, str) or not tweet:
            return np.zeros(vector_size)
        words = tweet.split()  # Tokenize by whitespace
        word_vectors = [model[word] for word in words if word in model]
        if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
            return np.zeros(vector_size)
        return np.mean(word_vectors, axis=0)

    # Apply preprocessing to each tweet and obtain vectors
    vector_size = 200  # Adjust based on the chosen GloVe model
    tweet_vectors_train = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in
                                     tqdm(df_train['Tweet'], desc="Generating Vectors")])
    tweet_df_train = pd.DataFrame(tweet_vectors_train)
    tweet_vectors_eval = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in
                                    tqdm(df_eval['Tweet'], desc="Generating Vectors")])
    tweet_df_eval = pd.DataFrame(tweet_vectors_eval)

    # Drop the columns that are not useful anymore
    df_train = df_train.drop(columns=['Timestamp', 'Tweet'])
    df_eval = df_eval.drop(columns=['Timestamp', 'Tweet'])
    # Attach the vectors into the original dataframe
    df_train = pd.concat([df_train, tweet_df_train], axis=1)
    df_eval = pd.concat([df_eval, tweet_df_eval], axis=1)
    # Group the tweets into their corresponding periods
    df_train = df_train.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    df_eval = df_eval.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    return df_train, df_eval
