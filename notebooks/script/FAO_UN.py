#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize': (15, 8)})
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random as rd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot
import warnings


shortlist_coutries = [
    'Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'Colombia', 'Egypt',
    'Ethiopia', 'France', 'Germany', 'India', 'Ireland', 'Italy', 'Kenya',
    'Mexico', 'Nigeria', 'Russian Federation', 'Spain',
    'United Kingdom of Great Britain and Northern Ireland',
    'United States of America'
]

continents = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']

cont_region = [
    'Eastern Africa', 'Middle Africa', 'Northern Africa', 'Southern Africa',
    'Western Africa', 'Northern America', 'Central America', 'Caribbean',
    'South America', 'Central Asia', 'Eastern Asia', 'Southern Asia',
    'South-eastern Asia', 'Western Asia', 'Eastern Europe', 'Northern Europe',
    'Southern Europe', 'Western Europe', 'Oceania'
]


vp_beef = ['Meat indigenous, cattle', 'Meat, cattle']


vp = pd.read_csv("./../temp/value-of-production.csv", encoding='latin1')
vp = vp.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'],
             axis=1)
vp = vp.loc[vp.Area.isin(shortlist_coutries)]
vp = vp.loc[vp.Item.isin(vp_beef)]
vp = vp.sort_values(by=["Area", "Item", "Year"])
vp.reset_index(inplace=True, drop=True)
vp

#SLC standing for Standard Local Currency will require conversion.


trade_cl_beef = ['Meat, beef and veal sausages', 'Meat, beef, preparations', 'Meat, cattle',
       'Meat, cattle, boneless (beef & veal)']


trade_cl = pd.read_csv("./../temp/trade-crops-livestock.csv",
                       encoding='latin1')
trade_cl = trade_cl.drop(
    ['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
trade_cl = trade_cl.loc[trade_cl.Area.isin(shortlist_coutries)]
trade_cl = trade_cl.loc[trade_cl.Item.isin(trade_cl_beef)]
trade_cl = trade_cl.sort_values(by=["Area", "Item", "Year"])
trade_cl.reset_index(inplace=True, drop=True)
trade_cl


trade_cl.Unit.unique()


trade_cl.Item.unique()


trade_cl.Element.unique()


trade_cl.isnull().sum()


# # SUA_CROPS_LIVE

sua_beef = ['Meat, cattle']

#sua_beef = ['Beef and Buffalo Meat', 'Beef Mutton Pigmeat Prim''Meat, beef and veal sausages',
#'Meat, beef, dried, salted, smoked', 'Meat, beef, preparations', 'Meat, cattle',
#'Meat, cattle, boneless (beef & veal)', 'Beef and Buffalo Meat', 'Beef Mutton Pigmeat Prim']


sua = pd.read_csv("SUA_Crops_Livestock.csv", encoding='latin1')
sua = sua.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'],
               axis=1)
sua = sua.loc[sua.Area.isin(shortlist_coutries)]
sua = sua.loc[sua.Item.isin(sua_beef)]
sua = sua.sort_values(by=["Area", "Item", "Year"])
sua.reset_index(inplace=True, drop=True)
sua


categories = ['Production', 'Import Quantity', 'Export Quantity']
sua_prod_imp_exp = sua.loc[sua.Element.isin(categories)]
sua_prod_imp_exp.reset_index(inplace=True,drop=True)


sua_prod_imp_exp.Item.unique()


sua_filled_df = pd.DataFrame(columns=sua_prod_imp_exp.columns)

for country in tqdm(list(sua_prod_imp_exp.Area.unique())):
    for type_of_meat in list(sua_prod_imp_exp.Item.unique()):
        for element in list(sua_prod_imp_exp.Element.unique()):
            temp_df = sua_prod_imp_exp.loc[
                (sua_prod_imp_exp.Area == country)
                & (sua_prod_imp_exp.Item == type_of_meat) &
                (sua_prod_imp_exp.Element == element)]
            temp_df.fillna(temp_df.Value.mean(), inplace=True)
            sua_filled_df = sua_filled_df.append(temp_df)
            sua_filled_df.reset_index(inplace=True, drop=True)

sua_filled_df.dropna(inplace=True)
sua_filled_df.isnull().sum()
sua_filled_df.reset_index(inplace=True, drop=True)


sua_filled_df


sua_filled_df["1000 Tonnes"] = sua_filled_df["Value"] / 1000
sua_filled_df


def plot(df, element, region, meat_type="Meat, Cattle"):
    df = df.loc[(df.Element == element) & (df.Area == region)]
    df.reset_index(inplace=True, drop=True)
    sns.set(rc={'figure.figsize': (25, 12)})  #Set figure size
    sns.lineplot(x='Year', y='1000 Tonnes',
                 data=df[["Year", "1000 Tonnes"]]).set_title(
                     "Region: {0} - Trade: {1} (Thousand Tonnes).".format(
                         region, meat_type))  #Plot
    plt.show()


for region in sua_filled_df.Area.unique():
    for element in sua_filled_df.Element.unique():
        plot(sua_filled_df, element, region)

        #Production, Import and Export, respectively. Each Country.


len(sua_filled_df.Element.unique()) * len(sua_filled_df.Area.unique())


#Distribution of Elements by Year by Region by item


sua.Item.unique()


sua.Element.unique()
#'Production', 'Import Quantity', 'Export Quantity'
#'Food supply (kcal/capita/day)', 'Food supply quantity (g/capita/day)', 'Protein supply quantity (g/capita/day)', 'Fat supply quantity (g/capita/day)'


sua.isnull().sum()


nutrientflow = ['Cropland nutrient flow']


env_soil_nut = pd.read_csv("Environment_Soil_nutrient_budget.csv",
                           encoding='latin1')
env_soil_nut = env_soil_nut.drop(
    ['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
env_soil_nut = env_soil_nut.loc[env_soil_nut.Area.isin(shortlist_coutries)]
env_soil_nut = env_soil_nut.sort_values(by=["Area", "Item", "Year"])
env_soil_nut.reset_index(inplace=True, drop=True)
env_soil_nut

#Unities according to Element
#Cropland nutrient flow per unit area: kg/ha
#Cropland nutrient flow: tonnes


env_soil_nut.Unit.unique()


env_soil_nut.Item.unique()


env_soil_nut.Element.unique()


env_soil_nut.isnull().sum()


#plt.figure(figsize=(18,14))
#sns.heatmap(env_soil_nut.corr(),annot=True,linecolor="white",lw=0.5)


env_soil_nut.dtypes


env_soil_nut.Item.unique()


env_soil_nut.Element.unique()


env_soil_nut.Year.unique()


env_soil_nut.Unit.unique()


env_soil_nut.Value.unique()


env_soil_nut.Flag.unique()

