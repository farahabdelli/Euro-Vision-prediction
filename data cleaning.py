
import csv 
import pandas as pd
import re
import sys
import nltk
import numpy as np

import matplotlib.pyplot as plt
from os import path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.core.dtypes.missing import isna

"""# **Data understanding**"""


# To store dataset in a Pandas Dataframe
ext2 = pd.read_csv('contestants.csv', encoding='UTF-8',sep=",", on_bad_lines='skip')
ext2



# To store dataset in a Pandas Dataframe
vote = pd.read_csv('votes.csv', encoding='UTF-8',sep=",", on_bad_lines='skip')
vote

"""# Contestants"""
#explore
def freq_finalist(df):
    effectifs = df.value_counts()
    modalites = effectifs.index # l'index de effectifs contient les modalités

    tab = pd.DataFrame(modalites, columns = ["final"]) # création du tableau à partir des modalités
    tab["Effectifs"] = effectifs.values
    tab["Fréquence"] = tab["Effectifs"] / 64*100 # len(data) renvoie la taille de l'échantillon
    sorted_tab=tab.sort_values(by=['Effectifs'], ascending=[False])[1:]
    fig = px.histogram(data_frame=sorted_tab[1:], 
             x="final", 
             y="Fréquence", 
             title="Fréquence d'être qualifié pour la finale entre 1956 et 2022",
             labels={ "final":"Pays", "Fréquence": "Fréquence"}
             )
    return sorted_tab,fig.show(renderer="colab")
df = ext2
df['final'] = np.where(df['running_final']>=1, df['to_country'], 0)
freq_finalist(df["final"])

#explore
df = df[df.final != 0]
years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
fig = px.bar(data_frame=df[df.year.isin(years)],
             x="final", 
             y="place_final", 
             title="Place finale des différents pays entre 2010 et 2022",
             labels={ "final":"Pays", "Fréquence": "Fréquence"},
             animation_frame="year",
             hover_name="to_country",
             )
fig.show(renderer="colab")

#missing values function
def missing_values(df):
    mis_val  = df.isnull().sum()
    mis_val_per  = df.isna().sum()/len(df)*100
    mis_val_table = pd.concat([mis_val, mis_val_per], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
          mis_val_table_ren_columns.iloc[:,:] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    return mis_val_table_ren_columns.tail(60)


"""# Votes"""
#explore
df1=vote
df1=df1.query("round == 'final'")
years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
fig = px.bar(data_frame=df1[df1.year.isin(years)],
             x="to_country", 
             y="total_points", 
             title="Place finale des différents pays entre 2010 et 2019",
             labels={ "final":"Pays", "Fréquence": "Fréquence"},
             animation_frame="year",
             hover_name="to_country"
             
             )

fig.show()
#fig.write_html("figvote.html")


"""# **Data preparation**"""

#data select function
def data_select(year,df,columns):
    df_select = df[df.year.isin(years)]
    df_select=df_select[columns]
    return df_select

#clean function : delete all rows with 'place_contest' > 26 and running_final = 0
def data_cleaning(df):
    indexNames = df[df['running_final'].isnull()].index
    df.drop(indexNames , inplace=True)
    return df

# apply select function on contestants dataset 
#select columns
columns = ["year", "to_country_id","to_country","performer","song","running_final","place_final","points_final"] 
years = [2016,2017,2018,2019,2021,2022] # select years
df_select = data_select(years,ext2,columns)
# apply clean function on contestants dataset
df_clean = data_cleaning(df_select)

#save to csv
df_clean.to_csv("contestants_clean.csv", sep="," ,encoding='UTF-8' )

# apply select function on votes dataset
df1=vote.query("round == 'final'")
columns = ["year","round", "from_country_id","to_country_id","from_country","to_country","total_points"] #select columns
years = [2016,2017,2018,2019,2021,2022] #select years
df_select_vote = data_select(years,df1,columns)
df_select_vote = df_select_vote.drop(['round'], axis=1)

#save to csv
df_select_vote.to_csv("vote_clean.csv", sep="," ,encoding='UTF-8' )

#Eurovision (1975-2022): Average points
df_average_points_per_country = df_select_vote.groupby('to_country_name').agg({'total_points': ['mean']}).sort_values(('total_points', 'mean'), ascending = False)
labels = df_average_points_per_country[('total_points', 'mean')].index
avg_points = list(df_average_points_per_country[('total_points', 'mean')])
plt.figure(figsize=(16,10))
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(labels, avg_points)
ax.invert_yaxis()
ax.set_ylabel('Countries')
ax.set_title('Eurovision (1975-2019): Average points')
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
fig.tight_layout()
fig.xaxis.set_tick_params(pad=5)
plt.show()

