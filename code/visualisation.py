from typing import List
import pandas  as pd
import plotly.express as px
import streamlit as st
import os
from Modélisation import preprocess_data, prediction
from sklearn.externals import joblib

# PATHS

DATAPATH = './sources/'
PLAYER_FILE = 'eurovision_players.csv'
TWEET_FILE = 'tweets_polarity.csv'  # to replace
EOF_PERIOD = 'eurovision_contests.csv'
ABS_DATAPATH = os.path.abspath(DATAPATH)
PREP_DATA = 'features.csv' 
RLOGIST = 'reg_logist.joblib'

# load data
players = pd.read_csv(os.path.join(ABS_DATAPATH, PLAYER_FILE), sep=';')
players = players.rename(columns={'to_country':'country'})

contest = pd.read_csv(os.path.join(ABS_DATAPATH, EOF_PERIOD), sep=',')
tweets = pd.read_csv(os.path.join(ABS_DATAPATH, TWEET_FILE), sep=',') 

data_model = pd.read_csv(os.path.join(ABS_DATAPATH, PREP_DATA), sep=';') 


def edition_player(edition, sel_ctry, players):
    if edition and sel_ctry:
        conditions = (players.year == edition) & (players.country.isin(sel_ctry))  
    elif edition:
        conditions = (players.year == edition)

    cond_players = players.loc[conditions, ['country', 'performer', 'song', 'running_final', 'place_final', 'points_final']]
    return cond_players

def winner_count(players):

    winners = players.loc[players['place_final'] == 1, 'country'].value_counts().reset_index().rename(columns={'index':'Country', 'country':'Number of edition won'})
    winners.sort_values(by='Number of edition won', inplace=True, ascending=False)
    win_plot = px.bar(winners, x="Country", y="Number of edition won")

    return win_plot

def point_evolution(players):

    points = players[['year', 'points_final']].groupby(by='year').max().reset_index()
    points = points.rename(columns={'points_final': 'Final points'})
    points.sort_values(by='year', ascending=False)
    evol_plot = px.line(points, x="year", y="Final points", title=None)

    return evol_plot

def get_data_prediction(edition, country):
    # get prep data for country and edition
    data = data_model.loc((data_model.country == country) & (data_model.year == edition))
    return data

def get_winner_prediction( data) :

    # load model 
    model = joblib.load(os.path.join(ABS_DATAPATH, RLOGIST))
    
    # get data prepare
    prep_data = preprocess_data(data)

    # predict
    pred = prediction (model , data, prep_data)

    return 'Nothing to show. Choose an edition to get the prediction.'

def prep_data_sentiment(tweets, edition):
    
    tweets.date = pd.to_datetime(tweets.date)
    ed_tweet = tweets.loc[(tweets.year == edition) & (tweets.date > (str(edition) + '-01-01')) & (tweets.date < (str(edition) +'-12-31')), ['country', 'polarity']]
    ed_tweet.loc[ed_tweet.polarity == 4 , 'sentiment'] = 'positif'
    ed_tweet.loc[ed_tweet.polarity == 0 , 'sentiment'] = 'negatif'
    ed_tweet = pd.get_dummies(ed_tweet, columns=['sentiment'], prefix=None)

    sentiments = (ed_tweet.groupby(by=['country'])[['sentiment_negatif', 'sentiment_positif']].sum()).reset_index()
    total = (sentiments.sentiment_negatif+sentiments.sentiment_positif)
    sentiments.sentiment_negatif = round(sentiments.sentiment_negatif*100 / total, 2)
    sentiments.sentiment_positif = round(sentiments.sentiment_positif*100 / total, 2)

    return sentiments

def get_sentiment(edition, country, tweets):

    if edition and country:
        sentiments = prep_data_sentiment(tweets, edition)
        return sentiments[sentiments.country.isin(country)]

    elif edition:
        sentiments = prep_data_sentiment(tweets, edition)
        return sentiments    
        
    else:
        return 'Nothing to show. Choose an edition and a country to get the sentiment.'


navs = st.sidebar.radio("Navigation", ["Home", "Prediction"])

if navs == "Home":

    head_ctn = st.container()
    head_ctn.title("Eurovision Winner Prediction")
    #st.image()
    head_ctn.header("Home")

    ########### player edition
    players_ctn = st.container()
    players_ctn.subheader("Finalist constestant by edition's year")

    # filter
    filt1, filt2 = players_ctn.columns(2) 
    editions = players.year.unique()
    countries = players.country.unique()

    sel_year = filt1.selectbox("Choose edition", list(editions), index=1)
    #edition = st.write(sel_year)
    sel_ctry = filt2.multiselect("Country", countries, default="Ukraine")

    # viz 1
    #st.dataframe(cond_players)
    players_ctn.table(edition_player(sel_year, sel_ctry, players))

    ########### winner_count
    winners_ctn = st.container()
    winners_ctn.subheader("Ranking of winning country by winning count")

    winners_ctn.plotly_chart(winner_count(players), use_container_width=True)

    ########### point evolution
    points_ctn = st.container()
    points_ctn.subheader("Evolution of the winning point over the year")
    #sel_year2 = st.selectbox("Choose edition", list(editions), index=1)
    points_ctn.plotly_chart(point_evolution(players))

if navs == "Prediction":
    head_ctn = st.container()
    head_ctn.title("Eurovision Winner Prediction")
    #st.image()
    head_ctn.header("Prediction")

    editions = players.year.unique()
    countries = players.country.unique()

    ########### prediction
    pred_ctn = st.container()
    pred_ctn.subheader("Prediction of the winner")
    # filter
    pd_sel_year = pred_ctn.selectbox("Choose edition which you want to predict the winner", list(editions), index=1)
    
    # result : winner, general sentiment and (prob)
    pred_result = get_winner_prediction(pd_sel_year)

    if isinstance(pred_result, str):
        pred_ctn.write(pred_result)
    else:
        pred_ctn.table(pred_result)


    ########### general sentiment of a contestant
    sentiment_ctn = st.container()
    sentiment_ctn.subheader("General sentiment of the public on the contestants")
    # filter
    filt1, filt2 = sentiment_ctn.columns(2) 
    
    sa_sel_year = filt1.selectbox("Choose edition", list(editions), index=1)
    #edition = st.write(sel_year)
    sa_sel_ctry = filt2.multiselect("Country", countries, default="Ukraine")

    # result : contestant, general sentiment 
    result = get_sentiment(sa_sel_year, sa_sel_ctry, tweets)

    if isinstance(result, str):
        sentiment_ctn.write(result)
    else:
        sentiment_ctn.table(result)
