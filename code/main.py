import pandas as pd

from polarity_analysis import (
    full_sentiment_training_pipeline,
    load_and_clean_sentiment_training,
    predict_tweets_polarity,
)
from tweets_cleaning import preprocess_tweets_content
from tweets_collection import full_tweets_retrieval_pipeline
from tweets_translation import translate_tweets
from utils import load_csv_data, load_json_data, load_model_data

if __name__ == "__main__":

    # ---------------- TWEETS COLLECTION ----------------#
    # Run the full tweets collection pipeline (25min)
    # tweets = full_tweets_retrieval_pipeline()

    # Or load the persisted collected tweets
    # tweets_fp = "../sources/tweets.json"
    # tweets = load_json_data(tweets_fp)

    # tweets_df = pd.DataFrame(tweets)

    # ---------------- TWEETS TRANSLATION ----------------#
    # Run the full tweets translation pipeline (26h)
    # translated_tweets_df = translate_tweets(tweets_df)

    # Or load the persisted translated tweets
    # translated_tweets_fp = "../sources/translated_tweets.csv"
    # translated_tweets_df = load_csv_data(translated_tweets_fp)

    # ---------------- TWEETS CLEANING ----------------#
    # cleaned_tweets_fp = "../sources/cleaned_tweets.csv"
    # Run the full tweets cleaning pipeline (1min)
    # cleaned_tweets = preprocess_tweets_content(translated_tweets_df, "translated_content", cleaned_tweets_fp)

    # Or load the persisted cleaned tweets
    # cleaned_tweets = load_csv_data(cleaned_tweets_fp)

    # ---------------- SENTIMENT TRAINING ----------------#
    ## --------------- Load & clean the dataset -------- ##
    # cleaned_sentiment_training_fp = "../sources/cleaned_sentiment_training.csv"
    # Run the full sentiment training dataset cleaning pipeline (5min)
    # cleaned_training_dataset = load_and_clean_sentiment_training(cleaned_sentiment_training_fp)

    # Or load the persisted cleaned sentiment training dataset
    # cleaned_training_dataset = load_csv_data(cleaned_sentiment_training_fp)

    ## --------------- Build & train the model --------- ##
    # sentiment_training_model_fp = "../sources/sentiment_model.pkl"
    # Run the full sentiment analysis model training pipeline (1h)
    """
    sentiment_analysis_model = full_sentiment_training_pipeline(
        cleaned_training_dataset,
        cleaned_tweets,
        'cleaned_content',
        sentiment_training_model_fp
    )
    """

    # Or load the persisted fitted model
    # sentiment_analysis_model = load_model_data(sentiment_training_model_fp)

    ## --------------- Generate polarity feature ------- ##
    # Run the polarity feature generation function
    tweets_w_polarity_fp = "../sources/tweets_polarity.csv"
    """
    tweets_w_polarity = predict_tweets_polarity(
        sentiment_analysis_model,
        cleaned_tweets,
        "cleaned_content",
        tweets_w_polarity_fp,
    )
    """
    # Or load the persisted tweets with polarity
    tweets_w_polarity = load_csv_data(tweets_w_polarity_fp)
