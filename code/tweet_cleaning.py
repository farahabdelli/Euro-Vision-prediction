import string
import time

import nltk
import pandas as pd
import regex as re
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


def shape_multiples_spaces(str_value):
    return str_value.strip()


def remove_punctuation(str_value):
    return str_value.translate(str.maketrans("", "", string.punctuation))


def remove_emojis(str_value, emoji_pattern):
    return emoji_pattern.sub(r"", str_value)


def shape_line_breaks(str_value):
    return str_value.replace("\n", " ")


def remove_urls(str_value):
    return re.sub("((www.[^s]+)|(https?://[^s]+))", " ", str_value)


def remove_numeric_chars(str_value):
    return re.sub("[0-9]+", "", str_value)


def remove_stop_words(str_value, stop_words):
    tokenized_str_value = str_value.split()
    cleaned_words_list = [w for w in tokenized_str_value if not w.lower() in stop_words]
    cleaned_str_value = " ".join([w for w in cleaned_words_list])
    return cleaned_str_value


def tokenize_content(str_value):
    return word_tokenize(str_value)


def stem_content(str_values, stemmer):
    return [stemmer.stem(str_value) for str_value in str_values]


def lemmatize_content(str_values, lemmatizer):
    return [lemmatizer.lemmatize(str_value) for str_value in str_values]


def preprocess_tweets_content(
    tweets_df, content_colname, cleaned_tweets_target_file_path
):
    start_time = time.time()

    stop_words = get_stop_words("en")
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )

    contents = tweets_df[content_colname]

    print("\n# Cleaning multiple spaces...")
    contents = contents.apply(lambda str_value: shape_multiples_spaces(str_value))

    print("\n# Cleaning emojis...")
    contents = contents.apply(lambda str_value: remove_emojis(str_value, emoji_pattern))

    print("\n# Cleaning urls...")
    contents = contents.apply(lambda str_value: remove_urls(str_value))

    print("\n# Cleaning numeric characters...")
    contents = contents.apply(lambda str_value: remove_numeric_chars(str_value))

    print("\n# Cleaning ponctuations...")
    contents = contents.apply(lambda str_value: remove_punctuation(str_value))

    print("\n# Cleaning line breaks...")
    contents = contents.apply(lambda str_value: shape_line_breaks(str_value))

    print("\n# Cleaning stop words...")
    contents = contents.apply(
        lambda str_value: remove_stop_words(str_value, stop_words)
    )

    print("\n# Tokenizing contents...")
    contents = contents.apply(lambda str_value: tokenize_content(str_value))

    print("\n# Stemming contents...")
    contents = contents.apply(lambda str_values: stem_content(str_values, stemmer))

    print("\n# Lemmatizing contents...")
    contents = contents.apply(
        lambda str_values: lemmatize_content(str_values, lemmatizer)
    )

    tweets_df.drop(columns=[content_colname], inplace=True)
    tweets_df["cleaned_content"] = contents

    print("\n# Persisting cleaned tweets...")
    tweets_df.to_csv(cleaned_tweets_target_file_path, index=False)

    print(f"\n-- Processing time : {(time.time() - start_time) / 60} minutes.")
