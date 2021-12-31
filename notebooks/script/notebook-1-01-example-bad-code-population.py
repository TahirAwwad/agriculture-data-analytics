#!/usr/bin/env python
# coding: utf-8

# Importing libraries and basic settings for ease of use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import altair as alt

get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')
sns.set_context("talk")

import warnings
warnings.filterwarnings("ignore") # I suppressed the warnings

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_mode_interactivity = "all"

#from pandas import read_csv
#from pandas import datetime
#from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
# pip install pmdarima
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA





# Read csv files into a pandas dataframe
#df_BDM = pd.read_csv("D:/BDM_Ireland1960_2021.csv")
#df_B   = pd.read_csv("D:/BirthsByCounty1985_2020.csv",sep="\t")
#df_D   = pd.read_csv("D:/deaths-region-2007-2020.csv")

#df_P   = pd.read_csv("D:/population_est_1950-2021.csv") # Commented out to remove bug


# Correction to make the program run
df_P   = pd.read_csv("./../assets/2021-12Dec-11-population-estimates-1950-2021-pea01.csv")


def missing_values(df):
    if (df.isnull().values.any()): print (df.isnull().sum()) # how many missing values?
    else: print("no missing values")

# Augmented Dickey-Fuller Test - ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print("Augmented Dickey-Fuller Test (ADF)\n==================================")
    print(f'T={series.count()}')
    print(f'- ADF the test statistic:     {result[0]}')
    print(f'- MacKinnons approx p-value:  {result[1]}')
    print(f'- number of lags used:        {result[2]}')

    print('\nCritial Values:\n---------------')
    for key, value in result[4].items():
        if (key=="10%"): print(f'{key}   {value}')
        else: print(f' {key}   {value}')
    print("\n")
# Kwiatkowski–Phillips–Schmidt–Shin Test - KPSS Test 
def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    print("Kwiatkowski–Phillips–Schmidt–Shin Test (KPSS)\n============================================")    
    print(f'T={series.count()}')
    print(f'- KPSS the test statistic:   {statistic}')
    print(f'- MacKinnons approx p-value: {p_value}')
    print(f'- number of lags used:       {n_lags}')

    print('\nCritical Values:\n---------------')
    for key, value in critical_values.items():
        if (key=="2.5%"): print(f'{key}   {value}')
        elif (key=="10%"): print(f' {key}   {value}')
        else: print(f'  {key}   {value}')
    print("\n")


# ==================================================


df_P.info # Structure and dimensions of the dataset


df_P.head() # first view of the dataset


# Dimension reduction:::
# Seems, Columns Statistic & UNIT cannot be used because they contain only one value.
# Examining the values in the Statistic column
pd.Series(df_P["Statistic"].unique()).sort_values()


# Examining the values in the UNIT column
pd.Series(df_P["UNIT"].unique()).sort_values()


# Only one value - cannot be used for analysis
# Deleting the Statistic column 
# Deleting the UNIT column 
df_P = df_P.drop(["Statistic"], axis='columns')
df_P = df_P.drop(["UNIT"], axis='columns')


# rename the columns
new_titles = {"Year":"Year", "Age Group":"Age", "Sex":"Sex", "VALUE":"population"}
df_P = df_P.rename(columns=new_titles)
df_P.info  # Structure and dimensions of the dataset


# any missing values?
missing_values(df_P)


df_P.head()


df_PO = df_P.drop(df_P[ (df_P["Age"] == "0 - 4 years") |
                        (df_P["Age"] == "0 - 14 years") |
                        (df_P["Age"] == "15 - 24 years") |
                        (df_P["Age"] == "15 years and over") |
                        (df_P["Age"] == "25 - 44 years") |
                        (df_P["Age"] == "45 - 64 years") |
                        (df_P["Age"] == "65 years and over") |
                        (df_P["Age"] == "All ages") ].index)
df_PO.info


df_PO = df_PO.reset_index(drop=True)
df_PO.info


missing_values(df_PO)


df_PP = df_PO


pd.Series(df_PP["population"].where(df_PP["Sex"] == "Both sexes").unique()).sort_values()


# Counting the "Both sexes" rows
df_PP.where(df_PP["Sex"] == "Both sexes").count(axis=0)


# "Both sexes" - not need for the population pyramids
# Deleting the "Both sexes" rows


df_PP = df_PP.drop(df_PP[(df_PP["Sex"] == "Both sexes")].index)
df_PP.info


# Counting the rows
df_PP.where(df_PP["Sex"] == "Both sexes").count(axis=0)


df_PP = df_PP.reset_index(drop=True)
df_PP.info


df_PP.where(df_PP["Age"] == "Under 1 year").count(axis=0)


# Examining the values in the Age Group column
pd.Series(df_PP["Age"].unique()) #.sort_values()
#df_PP["Age Group"].unique()


df_PP=df_PP.replace("Under 1 year"," 0-")
df_PP=df_PP.replace("1 - 4 years"," 1-")
df_PP=df_PP.replace("5 - 9 years"," 5-")
df_PP=df_PP.replace("10 - 14 years","10-")
df_PP=df_PP.replace("15 - 19 years","15-")
df_PP=df_PP.replace("20 - 24 years","20-")
df_PP=df_PP.replace("25 - 29 years","25-")
df_PP=df_PP.replace("30 - 34 years","30-")
df_PP=df_PP.replace("35 - 39 years","35-")
df_PP=df_PP.replace("40 - 44 years","40-")
df_PP=df_PP.replace("45 - 49 years","45-")
df_PP=df_PP.replace("50 - 54 years","50-")
df_PP=df_PP.replace("55 - 59 years","55-")
df_PP=df_PP.replace("60 - 64 years","60-")
df_PP=df_PP.replace("65 - 69 years","65-")
df_PP=df_PP.replace("70 - 74 years","70-")
df_PP=df_PP.replace("75 - 79 years","75-")
df_PP=df_PP.replace("80 - 84 years","80-")
df_PP=df_PP.replace("85 years and over","85-")
#df_PP.where(df_PP["Age Group"] == "Under 1 year")["Age Group"]="0"
#df_PP = df_PP.drop(df_PP[(df_PP["Sex"] == "Both sexes")].index)


df_PP = df_PP.reset_index(drop=True)
df_PP.info


df_PP.head()


# ========================================================================================
# interactive population pyramid code:


year_selector = alt.binding_range(min=1950, max=2021, step=1, name= "year selector")


select_year = alt.selection_single(name="population",fields=["Year"],
                                   bind=year_selector, init={"Year":2020} )


trunk = alt.Chart(df_PP, title="Age").add_selection(
    select_year).transform_filter(
    select_year).transform_calculate(
    gender=alt.datum.Sex).properties(width=300)


tree_color = alt.Scale(domain=['Male', 'Female'], range=['blue', 'violet'])


male_tree = trunk.transform_filter(alt.datum.gender == 'Male').encode(
    y=alt.Y('Age:O', axis=None, sort=alt.SortOrder('descending')),
    x=alt.X('sum(population):Q', scale=alt.Scale(domain=[0,300.0]),
    title='male population (1000)', sort=alt.SortOrder('descending')),
    color=alt.Color('gender:N', scale=tree_color)
    ).mark_bar().properties(title='Male')


y_trunk = trunk.encode(
    y=alt.Y('Age:O', axis=None, sort=alt.SortOrder('descending')),
        text=alt.Text('Age:O')).mark_text().properties(width=20)


female_tree = trunk.transform_filter(alt.datum.gender == 'Female').encode(
    y=alt.Y('Age:O', axis=None, sort=alt.SortOrder('descending')),
    x=alt.X('sum(population):Q', scale=alt.Scale(domain=[0,300.0]),
    title='female population (1000)'),
    color=alt.Color('gender:N', scale=tree_color)
    ).mark_bar().properties(title='Female')


alt.concat(male_tree, y_trunk, female_tree, spacing=2)




