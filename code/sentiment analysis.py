import pickle as pkl
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from tweets_cleaning import preprocess_tweets_content


def load_training_sentiment_analysis(target_file_path, dataset_colnames):
    return pd.read_csv(target_file_path, names=dataset_colnames)


def split_feature_target(training_dataset):
    X = training_dataset["cleaned_content"]
    y = training_dataset["target"]
    return X, y


def split_train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def vectorize_features(X_train, X_test, transfer_learning_test_set):
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(transfer_learning_test_set)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    return X_train, X_test


def build_lr_model_and_parameters():
    model = LogisticRegression()

    params = {
        "penalty": ["l2"],
        # "max_iter": [200, 400, 500, 700, 1000],
        "max_iter": [500],
        "multi_class": ["multinomial"],
    }

    return model, params


def tune_and_train_model(model, params, X_train, y_train):

    grid = GridSearchCV(model, params, verbose=2)
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def save_fitted_model(model, target_file_path):
    with open(target_file_path, "wb") as file:
        pkl.dump(model, file)


def load_and_clean_sentiment_training(clean_training_target_file_path):
    dataset_colnames = ["target", "ids", "date", "flag", "user", "text"]
    training_target_file_path = "../sources/sentiment_training.csv"

    print("\n# Loading external training dataset...")
    training_dataset = load_training_sentiment_analysis(
        training_target_file_path, dataset_colnames
    )

    print("\n# Cleaning training dataset...")
    cleaned_training_dataset = preprocess_tweets_content(
        training_dataset, "text", clean_training_target_file_path
    )

    return cleaned_training_dataset


def evaluated_sentiment_model(fitted_model, X_test, y_test):
    y_pred = fitted_model.predict(X_test)
    return classification_report(y_test, y_pred)


def full_sentiment_training_pipeline(
    cleaned_training_dataset,
    cleaned_tweets_df,
    content_colname,
    fitted_model_target_file_path,
):
    start_time = time.time()

    print("\n# Preprocessing training dataset...")
    X, y = split_feature_target(cleaned_training_dataset)
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.20)

    print("\n# Vectorizing features...")
    transfer_learning_test_set = cleaned_tweets_df[content_colname]
    X_train, X_test = vectorize_features(X_train, X_test, transfer_learning_test_set)

    print("\n# Buidling and fitting prediction model...")
    log_reg, params = build_lr_model_and_parameters()
    fitted_model = tune_and_train_model(log_reg, params, X_train, y_train)

    print("\n# Saving fitted model...")
    save_fitted_model(fitted_model, fitted_model_target_file_path)

    print("\n# Evaluating fitted model...")
    classification_report = evaluated_sentiment_model(fitted_model, X_test, y_test)
    print(classification_report)

    print(f"\n-- Processing time : {(time.time() - start_time) / 60} minutes.")

    return fitted_model


def predict_tweets_polarity(model, tweets_df, content_colname, polarity_df_fp):
    tweets_content = tweets_df[content_colname]

    print("\n# Vectorizing features...")
    tweets_content, tweets_content = vectorize_features(
        tweets_content, tweets_content, tweets_content
    )

    print("\n# Predicting tweets polarity...")
    polarities = model.predict(tweets_content)

    tweets_df["polarity"] = polarities
    print("\n# Persisting tweets with polarity...")
    tweets_df.to_csv(polarity_df_fp, index=False)
    return tweets_df
