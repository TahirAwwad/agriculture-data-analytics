#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(rc = {'figure.figsize':(15,8)})
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random as rd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
#from keras.models import Sequential
#from keras.layers import Dense
#import tensorflow as tf


shortlist_coutries = ['Argentina', 'Brazil', 'Canada', 
       'France', 'Germany', 'Ireland', 'Italy', 
       'Mexico', 'Spain',  
       'United Kingdom of Great Britain and Northern Ireland','United States of America']


# # Value of Production

vp_beef = ['Milk, whole fresh cow', 'Meat, cattle']


vp = pd.read_csv("Value_of_Production.csv", encoding='latin1')
vp = vp.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
vp = vp.loc[vp.Area.isin(shortlist_coutries)]
vp = vp.loc[vp.Item.isin(vp_beef)]
vp = vp.sort_values(by=["Area","Item", "Year"])
vp.reset_index(inplace=True,drop=True)
vp

#SLC standing for Standard Local Currency will require conversion.


# # Trade:
# ### Imports by Value and Quantity
# ### Exports by Value and Quantity

trade_cl_beef = ['Meat, cattle', 'Milk, whole fresh cow']


trade_cl = pd.read_csv("Trade_CropsLivestock.csv", encoding='latin1')
trade_cl = trade_cl.loc[trade_cl.Year >= 1990]
trade_cl = trade_cl.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
trade_cl = trade_cl.loc[trade_cl.Area.isin(shortlist_coutries)]
trade_cl = trade_cl.loc[trade_cl.Item.isin(trade_cl_beef)]
trade_cl = trade_cl.sort_values(by=["Area","Item", "Year"])
trade_cl.reset_index(inplace=True,drop=True)
trade_cl


trade_cl_quantity = trade_cl.loc[(trade_cl.Element == "Export Quantity")|(trade_cl.Element == "Import Quantity")]
trade_cl_value = trade_cl.loc[(trade_cl.Element == "Export Value")|(trade_cl.Element == "Import Value")]
trade_cl_quantity.reset_index(inplace=True,drop=True)
trade_cl_value.reset_index(inplace=True,drop=True)


years = range(1961,2021) #Year to analyse


#Creation of missing years to quantity
for c in trade_cl_quantity.Area.unique():
  country = trade_cl_quantity.loc[trade_cl_quantity.Area == c]
  country.reset_index(inplace=True,drop=True)
  for item in country.Item.unique():
    for year in years:
      country_fitered = country.loc[(country.Year == year) & (country.Item == item)]
      country_filt_import = country_fitered.loc[country_fitered.Element == "Import Quantity"]
      country_filt_import.reset_index(inplace=True,drop=True)
      if len(country_filt_import) < 1:
        series = country.loc[0]
        series["Item"] = item
        series["Element"] = "Import Quantity"
        series["Year"] = year
        series["Value"] = np.nan
        trade_cl_quantity=trade_cl_quantity.append(series)
      
      country_filt_export = country_fitered.loc[country_fitered.Element == "Export Quantity"]
      country_filt_export.reset_index(inplace=True,drop=True)
      if len(country_filt_export) < 1:
        series = country.loc[0]
        series["Item"] = item
        series["Element"] = "Export Quantity"
        series["Year"] = year
        series["Value"] = np.nan
        trade_cl_quantity=trade_cl_quantity.append(series)


for c in trade_cl_quantity.Area.unique():
  total = trade_cl_quantity.loc[trade_cl_quantity.Area == c].shape[0]
  n = trade_cl_quantity.loc[(trade_cl_quantity.Value.isnull()) & (trade_cl_quantity.Area == c)]
  nulls = n.shape[0]
  print("Percentage total of null for country {0} is {1}".format(c,round((nulls/total)*100,2)))
  print("\nMissing items:",n.Item.value_counts())
  print("\n\n")


#Creation of missing years to Value
for c in trade_cl_value.Area.unique():
  country = trade_cl_value.loc[trade_cl_value.Area == c] #Filter by country
  country.reset_index(inplace=True,drop=True) #Reset Index
  for item in country.Item.unique():
    for year in years:
      country_fitered = country.loc[(country.Year == year) & (country.Item == item)]
      country_filt_import = country_fitered.loc[country_fitered.Element == "Import Value"]
      country_filt_import.reset_index(inplace=True,drop=True)
      if len(country_filt_import) < 1:
        series = country.loc[0]
        series["Item"] = item
        series["Element"] = "Import Value"
        series["Year"] = year
        series["Value"] = np.nan
        trade_cl_value=trade_cl_value.append(series)
      
      country_filt_export = country_fitered.loc[country_fitered.Element == "Export Value"]
      country_filt_export.reset_index(inplace=True,drop=True)
      if len(country_filt_export) < 1:
        series = country.loc[0]
        series["Item"] = item
        series["Element"] = "Export Value"
        series["Year"] = year
        series["Value"] = np.nan
        trade_cl_value=trade_cl_value.append(series)


for c in trade_cl_value.Area.unique():
  total = trade_cl_value.loc[trade_cl_value.Area == c].shape[0]
  n = trade_cl_value.loc[(trade_cl_value.Value.isnull()) & (trade_cl_value.Area == c)]
  nulls = n.shape[0]
  print("Percentage total of null for country {0} is {1}".format(c,round((nulls/total)*100,2)))
  print("\nMissing items:",n.Item.value_counts())
  print("\n\n")


trade_cl_qt_filled = pd.DataFrame(columns=trade_cl_quantity.columns)
for c in trade_cl_quantity.Area.unique():
  for item in trade_cl_quantity.Item.unique():
    for element in trade_cl_quantity.Element.unique():
      country = trade_cl_quantity.loc[(trade_cl_quantity.Area == c)&(trade_cl_quantity.Item == item)& (trade_cl_quantity.Element == element )]
      country = country.sort_values(by="Year")#.interpolate()
      mean = round(country.Value.mean()) #Fill with mean
      country = country.fillna(mean)
      trade_cl_qt_filled = trade_cl_qt_filled.append(country)
      trade_cl_qt_filled.reset_index(drop=True,inplace=True)


trade_cl_vl_filled = pd.DataFrame(columns=trade_cl_value.columns)
for c in trade_cl_value.Area.unique():
  for item in trade_cl_value.Item.unique():
    for element in trade_cl_value.Element.unique():
      country = trade_cl_value.loc[(trade_cl_value.Area == c)&(trade_cl_value.Item == item)& (trade_cl_value.Element == element )]
      country = country.sort_values(by="Year")#.interpolate()
      mean = round(country.Value.mean()) #Fill with mean
      country = country.fillna(mean)
      trade_cl_vl_filled = trade_cl_vl_filled.append(country)
      trade_cl_vl_filled.reset_index(drop=True,inplace=True)


#Value in dolars
trade_cl_vl_filled["Total_value"] = trade_cl_vl_filled.Value * 1000
trade_cl_qt_filled["Total_value"] = trade_cl_qt_filled.Value * 1000


trade_cl_vl_filled.to_csv(r'/Users/robertobruxel/Desktop/fao/trade_cl_vl.csv', index = False)
trade_cl_qt_filled.to_csv(r'/Users/robertobruxel/Desktop/fao/trade_cl_qt.csv', index = False)


# # Land Use

land_use_item = ['Country area', 'Land area', 'Agriculture']

land_use = pd.read_csv('Inputs_LandUse_E_All_Data.csv', encoding='latin1')
land_use = land_use.loc[land_use.Item.isin(land_use_item)]
land_use = land_use.loc[land_use.Area.isin(shortlist_coutries)]
land_use = land_use.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
land_use.reset_index(inplace=True,drop=True)
land_use


land_use = pd.read_csv('Inputs_LandUse_E_All_Data.csv', encoding='latin1')
land_use = land_use.loc[land_use.Item == "Agriculture"]


# # Production

prod_item = ['Meat, cattle', 'Milk, whole fresh cow']

prod = pd.read_csv('Production_Crops_Livestock_E_All_Data.csv', encoding='latin1')
prod = prod.loc[prod.Item.isin(prod_item)]
prod = prod.loc[prod.Area.isin(shortlist_coutries)]
prod = prod.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
prod.reset_index(inplace=True,drop=True)
prod


prod_processed = prod.loc[(prod.Element == "Production")&(prod.Year != 2020)]
prod_processed.Element.unique()


prod_processed


# # Merge Datasets

#Dataframe to hold the values
total_df = prod_processed.sort_values(by=["Area","Item","Year"])[["Area","Item","Year"]]
total_df


#Create the dataframes to merge by columns filtering trade_cl dataframe by Element
trade_import_qt = trade_cl_qt_filled.loc[(trade_cl_qt_filled.Element == "Import Quantity")&(trade_cl_qt_filled.Year != 2020)]
trade_export_qt = trade_cl_qt_filled.loc[(trade_cl_qt_filled.Element == "Export Quantity")&(trade_cl_qt_filled.Year != 2020)]
trade_import_vl = trade_cl_vl_filled.loc[(trade_cl_vl_filled.Element == "Import Value")&(trade_cl_vl_filled.Year != 2020)]
trade_export_vl = trade_cl_vl_filled.loc[(trade_cl_vl_filled.Element == "Export Value")&(trade_cl_vl_filled.Year != 2020)]


#Merge all the Values from dataframes into just one
total_df = pd.merge(total_df,prod_processed[["Area","Item","Year","Value"]],how="left",left_on=['Area','Item',"Year"],right_on=['Area','Item',"Year"])
total_df = pd.merge(total_df,trade_import_qt[["Area","Item","Year","Value"]],how="left",left_on=['Area','Item',"Year"],right_on=['Area','Item',"Year"])
total_df = pd.merge(total_df,trade_export_qt[["Area","Item","Year","Value"]],how="left",left_on=['Area','Item',"Year"],right_on=['Area','Item',"Year"])
total_df = pd.merge(total_df,trade_import_vl[["Area","Item","Year","Value"]],how="left",left_on=['Area','Item',"Year"],right_on=['Area','Item',"Year"])
total_df = pd.merge(total_df,trade_export_vl[["Area","Item","Year","Value"]],how="left",left_on=['Area','Item',"Year"],right_on=['Area','Item',"Year"])
total_df = pd.merge(total_df,land_use[["Area","Year","Value"]],how="left",left_on=['Area',"Year"],right_on=['Area',"Year"])


#Rename Columns
total_df.columns = ["Area","Item","Year","Production_ton","Import_Qt_ton","Export_Qt_ton","Import_Vl_1000$","Export_Vl_1000$","Agri_1000ha"]
total_df


total_df.to_csv(r'/Users/robertobruxel/Desktop/fao/combined.csv', index = False)


ratio_df = pd.DataFrame()
#Area	Item	Year
ratio_df["Area"] = total_df["Area"] 
ratio_df["Item"] = total_df["Item"] 
ratio_df["Year"] = total_df["Year"] 

ratio_df["Production_ton"] = total_df["Production_ton"] / total_df["Agri_1000ha"]
ratio_df["Import_Qt_ton"] = total_df["Import_Qt_ton"] / total_df["Agri_1000ha"]
ratio_df["Export_Qt_ton"] = total_df["Export_Qt_ton"] / total_df["Agri_1000ha"]
ratio_df["Import_Vl_1000$"] = total_df["Import_Vl_1000$"] / total_df["Agri_1000ha"]
ratio_df["Export_Vl_1000$"] = total_df["Export_Vl_1000$"] / total_df["Agri_1000ha"]
ratio_df


ratio_df.to_csv(r'/Users/robertobruxel/Desktop/fao/combined_ratio.csv', index = False)




