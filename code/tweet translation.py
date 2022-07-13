import time

import pandas as pd
from googletrans import Translator


def translate_single_content(str_value, translator):
    translation = translator.translate(str_value, sleep_seconds=10)
    return translation.text


def translate_multiple_content(sentences, translator):

    translations = []
    not_translated_index = []
    not_translated_sentences = []

    for index, sentence in enumerate(sentences):
        print(f"\tTranslation {index+1}/{len(sentences)}...")
        try:
            translations.append(translate_single_content(sentence, translator))
            time.sleep(1)
        except:
            print(f"\tSentence {index+1} not translated...")
            translations.append(sentence)
            not_translated_index.append(index)
            not_translated_sentences.append(sentence)
            pass

    return translations, not_translated_index


def translate_tweets(tweets_df):
    start_time = time.time()
    translated_tweets_target_file_path = "../sources/translated_tweets.csv"

    contents = tweets_df["content"]
    translator = Translator()
    print("\n# Translating tweets contents...")
    translated_contents, not_translated_index = translate_multiple_content(
        contents.values.tolist(), translator
    )
    print(f"\tnot_translated_sentences_count : {len(not_translated_index)}")
    print(f"\tnot_translated_sentences_index : {not_translated_index}")
    tweets_df["translated_content"] = translated_contents
    tweets_df = tweets_df.drop(columns=["content"])

    print("\n# Persisting translated tweets...")
    tweets_df.to_csv(translated_tweets_target_file_path, index=False)

    print(f"\n-- Processing time : {(time.time() - start_time) / 60} minutes.")

    return tweets_df
