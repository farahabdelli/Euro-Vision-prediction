import json
import time

import pandas as pd
import snscrape.modules.twitter as sntwitter


def load_and_preprocess_data(players_data_file_path, contests_data_file_path):
    pd.set_option("mode.chained_assignment", None)
    players_df_full = pd.read_csv(players_data_file_path, delimiter=";")
    players_df = players_df_full[
        ["year", "to_country", "performer", "song", "place_final", "points_final"]
    ]
    players_df.loc[players_df.place_final, "winner"] = players_df.place_final == 1
    players_df = players_df.astype({"year": "str"})

    contests_df = pd.read_csv(contests_data_file_path)
    contests_df = contests_df.astype(
        {"year": "str", "start_date": "str", "end_date": "str"}
    )

    return players_df, contests_df


def format_field_value(field_value):
    unauthorized_chars = [
        "&",
        "{",
        "(",
        "[",
        "-",
        "|",
        "~",
        "`",
        "\\",
        "_",
        "^",
        ")",
        "]",
        "+",
        "=",
        "}",
        "?",
        "!",
        ",",
        ";",
        ".",
        "/",
        ":",
        "*",
        "%",
    ]

    for char in unauthorized_chars:
        field_value = field_value.replace(char, "")
        field_value = field_value.strip()

    field_value.replace("United KingdomUK", "United Kingdom")
    return field_value


def build_query_str(input_row, start_date, end_date):

    hashtags_bases = ["#EUROVISION", "#eurovision", "#EVSC", "#evsc"]
    fields_names_to_process = ["to_country", "performer", "song"]

    query_str = "("
    for base in hashtags_bases:
        query_str += base + input_row["year"]
        if base != hashtags_bases[len(hashtags_bases) - 1]:
            query_str += " OR "

    query_str += ") ("

    for field_name, field_value in zip(input_row.index, input_row):
        if field_name in fields_names_to_process:
            field_value = format_field_value(field_value)
            field_values = field_value.split()

            if field_name == "performer":
                for value in field_values:
                    query_str += value
                    query_str += " OR #" + value
                    # query_str += value
                    if value != field_values[len(field_values) - 1]:
                        query_str += " OR "

            elif field_name == "song" or field_name == "to_country":
                if len(field_values) == 1:
                    query_str += field_value
                    query_str += " OR #" + field_value
                    # query_str += '#' +field_value
                else:
                    joined_field_values = "".join(field_value.split())
                    query_str += "#" + joined_field_values

            if field_name != input_row.index[len(input_row.index) - 1]:
                query_str += " OR "
    query_str += ")"

    query_parameters = (
        f" until:{end_date} since:{start_date} -filter:links -filter:replies"
    )
    query_str += query_parameters

    return query_str


def build_queries(players_data, contests_data, years_to_search):
    queries = []

    for _, row in players_data.iterrows():
        if row["year"] in years_to_search:
            query = {}
            query["year"] = row["year"]
            query["country"] = row["to_country"]
            query["points_final"] = row["points_final"]
            query["place_final"] = row["place_final"]
            query["winner"] = row["winner"]

            filtered_contests = contests_data[contests_data["year"] == row["year"]]
            start_date = filtered_contests["start_date"].iloc[0]
            end_date = filtered_contests["end_date"].iloc[0]

            query["query_str"] = build_query_str(row, start_date, end_date)
            queries.append(query)

    return queries


def process_queries(queries):
    queries_results = []
    max_nb_tweets = 500
    null_queries_count = 0
    null_queries = []

    for query_index, query in enumerate(queries):
        print(f"\tProcessing query {query_index + 1} / {len(queries)}...")
        nb_results = 0
        query_results = sntwitter.TwitterSearchScraper(query["query_str"]).get_items()
        for tweet_index, tweet in enumerate(query_results):
            if tweet_index == max_nb_tweets:
                break
            else:
                queries_results.append(
                    {
                        "year": query["year"],
                        "country": query["country"],
                        "points_final": query["points_final"],
                        "place_final": query["place_final"],
                        "winner": query["winner"],
                        "tweet": tweet,
                    }
                )
                nb_results += 1

        print(f"\t{nb_results} tweets retrieved")
        if nb_results == 0:
            null_queries_count += 1
            null_queries.append(query)

    return queries_results, null_queries_count, null_queries


def manage_tweets_objects(queries_results):
    tweets = []
    for tweet_object in queries_results:
        tweet = {}
        tweet["year"] = tweet_object["year"]
        tweet["country"] = tweet_object["country"]
        tweet["points_final"] = tweet_object["points_final"]
        tweet["place_final"] = tweet_object["place_final"]
        tweet["winner"] = tweet_object["winner"]

        tweet_content = tweet_object["tweet"]
        tweet["date"] = tweet_content.date
        tweet["content"] = tweet_content.content
        tweet["hashtags"] = tweet_content.hashtags
        tweet["retweet_count"] = tweet_content.retweetCount
        tweet["like_count"] = tweet_content.likeCount
        tweet["user_username"] = tweet_content.user.username
        tweet["user_verified"] = tweet_content.user.verified
        tweet["user_followers_count"] = tweet_content.user.followersCount
        tweets.append(tweet)
    return tweets


def persist_collected_tweets(data, target_file_path):
    with open(target_file_path, "w") as outfile:
        json.dump(data, outfile, indent=4, default=str)


def full_tweets_retrieval_pipeline():

    start_time = time.time()

    players_data_file_path = "../sources/eurovision_players.csv"
    contests_data_file_path = "../sources/eurovision_contests.csv"
    # queries_results_target_file_path = "../sources/queries_results.json"
    tweets_target_file_path = "../sources/tweets.json"

    # Raw data retrieval and preprocessing
    print("\n# Loading and preprocessing data...")
    players_df, contests_df = load_and_preprocess_data(
        players_data_file_path, contests_data_file_path
    )

    # Dataset existing years
    print("\n# Managing datasets dates...")
    years_players = players_df["year"].unique().tolist()
    years_contests = contests_df["year"].unique().tolist()
    # Shared years between players and contests dataset
    years_to_search = [year for year in years_players if year in years_contests]

    # Queries building
    print("\n# Building Twitter advanced queries...")
    queries = build_queries(players_df, contests_df, years_to_search)
    # queries = queries[:10]

    # Processing queries on Twitter API
    print("\n# Processing queries...")
    queries_results, null_queries_count, null_queries = process_queries(queries)
    print(f"\tnull_queries_count : {null_queries_count}")
    print(f"\tnull_queries : {null_queries}")

    # Generate tweets dictionnaries from snscrapper tweet objects
    print("\n# Customizing tweets from queries results...")
    tweets = manage_tweets_objects(queries_results)

    # Tweets persisting under JSON format file
    print("\n# Persisting customized tweets...")
    persist_collected_tweets(tweets, tweets_target_file_path)

    print(f"\n-- Processing time : {(time.time() - start_time) / 60} minutes.")

    return tweets
