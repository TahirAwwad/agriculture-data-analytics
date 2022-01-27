#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from data_analytics.graphs import display_caption
from datetime import datetime
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm
import data_analytics.zip_file as zip_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import warnings

sns.set(rc={'figure.figsize': (15, 8)})
warnings.filterwarnings('ignore')


shortlist_coutries = [
    'Argentina', 'Brazil', 'Canada', 'France', 'Germany', 'Ireland', 'Italy',
    'Mexico', 'Spain', 'United Kingdom of Great Britain and Northern Ireland',
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


# # Value of Production

vp_beef = ['Meat, cattle']


filepath = './../temp/value-of-production.csv'
zip_filepath = './../assets/value-of-production.zip'
temp_directory = './../temp'
zip_file.unzip_required_asset(filepath, zip_filepath, temp_directory)

vp = pd.read_csv(filepath, encoding='latin1')
vp = vp.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'],
             axis=1)
vp = vp.loc[vp.Area.isin(shortlist_coutries)]
vp = vp.loc[vp.Item.isin(vp_beef)]
vp = vp.sort_values(by=["Area", "Item", "Year"])
vp.reset_index(inplace=True, drop=True)
vp

#SLC standing for Standard Local Currency will require conversion.


# # Trade:
# Imports by Value and Quantity & Exports by Value and Quantity

trade_cl_beef = ['Meat, cattle']


filepath = './../temp/trade-crops-livestock.csv'
zip_filepath = './../assets/trade-crops-livestock/'
temp_directory = './../temp'
zip_file.unzip_required_asset(filepath, zip_filepath, temp_directory)

trade_cl = pd.read_csv(filepath, encoding="latin1")

#trade_cl = trade_cl.loc[trade_cl.Year >= 1990]
trade_cl = trade_cl.drop(
    ['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
trade_cl = trade_cl.loc[trade_cl.Area.isin(shortlist_coutries)]
trade_cl = trade_cl.loc[trade_cl.Item.isin(trade_cl_beef)]
trade_cl = trade_cl.sort_values(by=["Area", "Item", "Year"])
trade_cl.reset_index(inplace=True, drop=True)
trade_cl


trade_cl_quantity = trade_cl.loc[(trade_cl.Element == "Export Quantity") |
                                 (trade_cl.Element == "Import Quantity")]
trade_cl_value = trade_cl.loc[(trade_cl.Element == "Export Value") |
                              (trade_cl.Element == "Import Value")]
trade_cl_quantity.reset_index(inplace=True, drop=True)
trade_cl_value.reset_index(inplace=True, drop=True)


years = range(1961,2021) #Year to analyse


#Creation of missing years to quantity
for c in trade_cl_quantity.Area.unique():
    country = trade_cl_quantity.loc[trade_cl_quantity.Area == c]
    country.reset_index(inplace=True, drop=True)
    for item in country.Item.unique():
        for year in years:
            country_fitered = country.loc[(country.Year == year)
                                          & (country.Item == item)]
            country_filt_import = country_fitered.loc[country_fitered.Element
                                                      == "Import Quantity"]
            country_filt_import.reset_index(inplace=True, drop=True)
            if len(country_filt_import) < 1:
                series = country.loc[0]
                series["Item"] = item
                series["Element"] = "Import Quantity"
                series["Year"] = year
                series["Value"] = np.nan
                trade_cl_quantity = trade_cl_quantity.append(series)

            country_filt_export = country_fitered.loc[country_fitered.Element
                                                      == "Export Quantity"]
            country_filt_export.reset_index(inplace=True, drop=True)
            if len(country_filt_export) < 1:
                series = country.loc[0]
                series["Item"] = item
                series["Element"] = "Export Quantity"
                series["Year"] = year
                series["Value"] = np.nan
                trade_cl_quantity = trade_cl_quantity.append(series)


for c in trade_cl_quantity.Area.unique():
    total = trade_cl_quantity.loc[trade_cl_quantity.Area == c].shape[0]
    n = trade_cl_quantity.loc[(trade_cl_quantity.Value.isnull())
                              & (trade_cl_quantity.Area == c)]
    nulls = n.shape[0]
    print("Percentage total of null for country {0} is {1}".format(
        c, round((nulls / total) * 100, 2)))
    print("\nMissing items:", n.Item.value_counts())
    print("\n\n")


#Creation of missing years to Value
for c in trade_cl_value.Area.unique():
    country = trade_cl_value.loc[trade_cl_value.Area == c]  #Filter by country
    country.reset_index(inplace=True, drop=True)  #Reset Index
    for item in country.Item.unique():
        for year in years:
            country_fitered = country.loc[(country.Year == year)
                                          & (country.Item == item)]
            country_filt_import = country_fitered.loc[country_fitered.Element
                                                      == "Import Value"]
            country_filt_import.reset_index(inplace=True, drop=True)
            if len(country_filt_import) < 1:
                series = country.loc[0]
                series["Item"] = item
                series["Element"] = "Import Value"
                series["Year"] = year
                series["Value"] = np.nan
                trade_cl_value = trade_cl_value.append(series)

            country_filt_export = country_fitered.loc[country_fitered.Element
                                                      == "Export Value"]
            country_filt_export.reset_index(inplace=True, drop=True)
            if len(country_filt_export) < 1:
                series = country.loc[0]
                series["Item"] = item
                series["Element"] = "Export Value"
                series["Year"] = year
                series["Value"] = np.nan
                trade_cl_value = trade_cl_value.append(series)


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
            country = trade_cl_quantity.loc[(trade_cl_quantity.Area == c)
                                            & (trade_cl_quantity.Item == item)
                                            & (trade_cl_quantity.Element
                                               == element)]
            country = country.sort_values(by="Year")  #.interpolate()
            mean = round(country.Value.mean())  #Fill with mean
            country = country.fillna(mean)
            trade_cl_qt_filled = trade_cl_qt_filled.append(country)
            trade_cl_qt_filled.reset_index(drop=True, inplace=True)


trade_cl_vl_filled = pd.DataFrame(columns=trade_cl_value.columns)
for c in trade_cl_value.Area.unique():
    for item in trade_cl_value.Item.unique():
        for element in trade_cl_value.Element.unique():
            country = trade_cl_value.loc[(trade_cl_value.Area == c)
                                         & (trade_cl_value.Item == item) &
                                         (trade_cl_value.Element == element)]
            country = country.sort_values(by="Year")  #.interpolate()
            mean = round(country.Value.mean())  #Fill with mean
            country = country.fillna(mean)
            trade_cl_vl_filled = trade_cl_vl_filled.append(country)
            trade_cl_vl_filled.reset_index(drop=True, inplace=True)


"""population = pd.read_excel("Population Sample Data.xlsx")
population.columns = ["COUNTRY","POPULATION NO.","POP per 1000"]
dic = {}
for i in range(len(population)):
  dic.update({
      population.COUNTRY[i]:population["POPULATION NO."][i]
  })
dic"""


#Value in dolars
trade_cl_vl_filled["Total_value"] = trade_cl_vl_filled.Value * 1000
#trade_cl_vl_filled.tail()
"""#Add population
pop = []
for i in range(len(trade_cl_vl_filled)):
  pop.append(int(dic[trade_cl_vl_filled.Area[i]].replace(",","")))
trade_cl_vl_filled["Population"] = pop

#Calculate Per Capita
trade_cl_vl_filled["Per Capita"] = trade_cl_vl_filled.Total_value / trade_cl_vl_filled.Population
trade_cl_vl_filled["Per Capita"] = trade_cl_vl_filled["Per Capita"].round(3)"""


"""for element in trade_cl_vl_filled.Element.unique():
  for item in trade_cl_vl_filled.Item.unique():
    #Filter by Trade Type and Product Type
    imp = trade_cl_vl_filled.loc[(trade_cl_vl_filled.Element == element)&(trade_cl_vl_filled.Item == item)]
    imp.reset_index(inplace=True,drop=True)
    sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size

    #Plot
    sns.lineplot(x='Year', y='Per Capita',hue="Area",
                data=imp[["Area","Year","Per Capita"]], color="red" ).set_title("{0} -  {1}".format(element,item)) #Plot
    plt.show()"""


#Value in dolars
trade_cl_qt_filled["Total_value"] = trade_cl_qt_filled.Value * 1000
#trade_cl_qt_filled.tail()
"""#Add population
pop = []
for i in range(len(trade_cl_qt_filled)):
  pop.append(int(dic[trade_cl_qt_filled.Area[i]].replace(",","")))
trade_cl_qt_filled["Population"] = pop

#Calculate Per Capita
trade_cl_qt_filled["Per Capita"] = trade_cl_qt_filled.Total_value / trade_cl_qt_filled.Population
trade_cl_qt_filled["Per Capita"] = trade_cl_qt_filled["Per Capita"].round(3)
trade_cl_qt_filled.head()"""


"""for element in trade_cl_qt_filled.Element.unique():
  for item in trade_cl_qt_filled.Item.unique():
    #Filter by Trade Type and Product Type
    imp = trade_cl_qt_filled.loc[(trade_cl_qt_filled.Element == element)&(trade_cl_qt_filled.Item == item)]
    imp.reset_index(inplace=True,drop=True)
    sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size

    #Plot
    sns.lineplot(x='Year', y='Per Capita',hue="Area",
                data=imp[["Area","Year","Per Capita"]], color="red" ).set_title("{0} -  {1}".format(element,item)) #Plot
    plt.show()"""


'''def plot(df,region,meat_type="Meat, Cattle"):
  df = df.loc[df.Area == region]
  df.reset_index(inplace=True,drop=True)
  sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size
  for element in df.Element.unique():
    sns.lineplot(x='Year', y='1000 Tonnes', data=df[["Year","1000 Tonnes"]]).set_title("Region: {0} - Trade: {1} (Thousand Tonnes).".format(region,meat_type)) #Plot
  plt.savefig("images/"+ region +'.png')
  plt.show()

for region in sua_filled_df.Area.unique():
  plot(sua_filled_df,region)'''

#Distribution of Elements by Year by Region by item


# # Land Use

land_use_item = ['Agriculture']

filepath = './../temp/inputs-land-use-e-all-data.csv'
zip_filepath = './../assets/inputs-land-use-e-all-data.zip'
temp_directory = './../temp'
zip_file.unzip_required_asset(filepath, zip_filepath, temp_directory)

land_use = pd.read_csv(filepath, encoding='latin1')
land_use = land_use.loc[land_use.Item.isin(land_use_item)]
land_use = land_use.loc[land_use.Area.isin(shortlist_coutries)]
land_use = land_use.drop(
    ['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
land_use.reset_index(inplace=True, drop=True)
land_use


# # Production

prod_item = ['Meat, cattle']

filepath = './../temp/production-crops-livestock-e-all-data.csv'
zip_filepath = './../assets/production-crops-livestock-e-all-data.zip'
temp_directory = './../temp'
zip_file.unzip_required_asset(filepath, zip_filepath, temp_directory)

prod = pd.read_csv(filepath, encoding='latin1')

prod = prod.loc[prod.Item.isin(prod_item)]
prod = prod.loc[prod.Area.isin(shortlist_coutries)]
prod = prod.drop(
    ['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
prod.reset_index(inplace=True, drop=True)
prod


prod_processed = prod.loc[(prod.Element == "Production") & (prod.Year != 2020)]
prod_processed.Element.unique()


prod_processed


# # Merge Datasets

#Dataframe to hold the values
total_df = prod_processed.sort_values(by=["Area", "Item", "Year"])[[
    "Area", "Item", "Year"
]]
total_df


#Create the dataframes to merge by columns filtering trade_cl dataframe by Element
trade_import_qt = trade_cl_qt_filled.loc[
    (trade_cl_qt_filled.Element == "Import Quantity")
    & (trade_cl_qt_filled.Year != 2020)]
trade_export_qt = trade_cl_qt_filled.loc[
    (trade_cl_qt_filled.Element == "Export Quantity")
    & (trade_cl_qt_filled.Year != 2020)]
trade_import_vl = trade_cl_vl_filled.loc[
    (trade_cl_vl_filled.Element == "Import Value")
    & (trade_cl_vl_filled.Year != 2020)]
trade_export_vl = trade_cl_vl_filled.loc[
    (trade_cl_vl_filled.Element == "Export Value")
    & (trade_cl_vl_filled.Year != 2020)]

#Merge all the Values from dataframes into just one
total_df = pd.merge(total_df,
                    prod_processed[["Area", "Item", "Year", "Value"]],
                    how="left",
                    left_on=['Area', 'Item', "Year"],
                    right_on=['Area', 'Item', "Year"])
total_df = pd.merge(total_df,
                    trade_import_qt[["Area", "Item", "Year", "Value"]],
                    how="left",
                    left_on=['Area', 'Item', "Year"],
                    right_on=['Area', 'Item', "Year"])
total_df = pd.merge(total_df,
                    trade_export_qt[["Area", "Item", "Year", "Value"]],
                    how="left",
                    left_on=['Area', 'Item', "Year"],
                    right_on=['Area', 'Item', "Year"])
total_df = pd.merge(total_df,
                    trade_import_vl[["Area", "Item", "Year", "Value"]],
                    how="left",
                    left_on=['Area', 'Item', "Year"],
                    right_on=['Area', 'Item', "Year"])
total_df = pd.merge(total_df,
                    trade_export_vl[["Area", "Item", "Year", "Value"]],
                    how="left",
                    left_on=['Area', 'Item', "Year"],
                    right_on=['Area', 'Item', "Year"])
total_df = pd.merge(total_df,
                    land_use[["Area", "Year", "Value"]],
                    how="left",
                    left_on=['Area', "Year"],
                    right_on=['Area', "Year"])

#Rename Columns
total_df.columns = [
    "Area", "Item", "Year", "Production_ton", "Import_Qt_ton", "Export_Qt_ton",
    "Import_Vl_1000$", "Export_Vl_1000$", "Agri_1000ha"
]
total_df


ratio_df = pd.DataFrame()
#Area	Item	Year
ratio_df["Area"] = total_df["Area"]
ratio_df["Item"] = total_df["Item"]
ratio_df["Year"] = total_df["Year"]

ratio_df[
    "Production_ton"] = total_df["Production_ton"] / total_df["Agri_1000ha"]
ratio_df["Import_Qt_ton"] = total_df["Import_Qt_ton"] / total_df["Agri_1000ha"]
ratio_df["Export_Qt_ton"] = total_df["Export_Qt_ton"] / total_df["Agri_1000ha"]
ratio_df[
    "Import_Vl_1000$"] = total_df["Import_Vl_1000$"] / total_df["Agri_1000ha"]
ratio_df[
    "Export_Vl_1000$"] = total_df["Export_Vl_1000$"] / total_df["Agri_1000ha"]
ratio_df


ratio_df.to_csv("./../artifacts/land-ratio-eda-output.csv")


# # Descriptive statistics

# ## Heatmap (Correlations)

for country in total_df.Area.unique():
    for item in total_df.Item.unique():
        country_df = total_df.loc[(total_df.Area == country)
                                  & (total_df.Item == item)]
        plt.figure(figsize=(18, 14))
        sns.heatmap(country_df.corr(), annot=True, linecolor="blue",
                    lw=0.5).set_title("Region: {0} - Item: {1}.".format(
                        country, item))
        plt.show()


# ## Boxplot (Distribution)

plt.figure(figsize=(20, 30))

for country in total_df.Area.unique():
    for item in total_df.Item.unique():
        country_df = total_df.loc[(total_df.Area == country)
                                  & (total_df.Item == item)]

        col_dict = {}
        contador = 1
        for column in country_df.drop(["Area", "Item", "Year"],
                                      axis=1).columns:
            col_dict.update({column: contador})
            contador += 1

        for variable, i in col_dict.items():
            t = "{0}\nRegion: {1} - Item: {2}.".format(variable, country, item)
            plt.subplot(2, 3, i)
            country_df.boxplot(column=variable)
            plt.title(t)
        plt.show()


# ## Histogram

for country in total_df.Area.unique():
    for item in total_df.Item.unique():
        country_df = total_df.loc[(total_df.Area == country)
                                  & (total_df.Item == item)]

        g = sns.PairGrid(country_df.drop(["Area", "Item", "Year"], axis=1))
        g.map_upper(sns.histplot)
        g.map_lower(sns.scatterplot)
        g.map_diag(sns.histplot, kde=True)
        g.add_legend(title="Region: {0}\nItem: {1}.".format(country, item))
        plt.show()


columns_to_plot = list(ratio_df.columns)
columns_to_plot.remove('Area')
columns_to_plot.remove('Item')
columns_to_plot.remove('Year')
columns_to_plot


columns = ['Area', 'Year']
columns.append("Production_ton")
ratio_df[columns]


for col in columns_to_plot:
    #Filter by Trade Type and Product Type
    imp = ratio_df.loc[(trade_cl_vl_filled.Element == element)]
    imp.reset_index(inplace=True, drop=True)
    columns = ['Area', 'Year']
    columns.append(col)

    sns.set(rc={'figure.figsize': (25, 12)})  #Set figure size

    #Plot
    sns.lineplot(
        x='Year', y=col,
        hue="Area", data=ratio_df[columns], color="red").set_title(
            "Meat Cattle - Ratio Agri_1000ha  -  {0}".format(col))  #Plot
    plt.show()

