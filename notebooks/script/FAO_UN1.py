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
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')


from pathlib import Path
from zipfile import ZipFile
import os


def join_split_zip_files(split_zip_files_directory: str, destination_dir: str = './temp/', zip_filename: str = 'temp.zip') -> None:
    split_zip_files = os.listdir(split_zip_files_directory)
    zip_filepath = os.path.join(destination_dir, zip_filename)
    if os.path.isfile(zip_filepath):
        os.remove(zip_filepath)
    for split_zip_file in split_zip_files:
        with open(zip_filepath, "ab") as zip_file:
            with open(os.path.join(split_zip_files_directory, split_zip_file), "rb") as split_zip:
                zip_file.write(split_zip.read())
    return zip_filepath


def unzip_file(zip_filepath: str, destination_dir: str) -> None:
    with ZipFile(zip_filepath, 'r') as zipfile:
        zipfile.extractall(destination_dir)


def unzip_required_asset(filepath: str, zip_path: str, destination_dir: str) -> None:
    if not os.path.isfile(filepath):
        if os.path.isfile(zip_path):
            unzip_file(zip_path, destination_dir)
        elif os.path.isdir(zip_path):
            zip_filepath = join_split_zip_files(zip_path, destination_dir)
            unzip_file(zip_filepath, destination_dir)
            os.remove(zip_filepath)


shortlist_coutries = ['Argentina', 'Brazil', 'Canada', 
       'France', 'Germany', 'Ireland', 'Italy', 
       'Mexico', 'Spain',  
       'United Kingdom of Great Britain and Northern Ireland','United States of America']

continents = ['Africa', 'Americas', 'Asia',
      'Europe', 'Oceania']

cont_region = ['Eastern Africa',
       'Middle Africa', 'Northern Africa', 'Southern Africa',
       'Western Africa', 'Northern America',
       'Central America', 'Caribbean', 'South America', 
       'Central Asia', 'Eastern Asia', 'Southern Asia',
       'South-eastern Asia', 'Western Asia', 'Eastern Europe',
       'Northern Europe', 'Southern Europe', 'Western Europe', 'Oceania']


# # Value of Production

vp_beef = ['Milk, whole fresh cow', 'Meat, cattle']


filepath = './../temp/value-of-production.csv'
zip_filepath = './../assets/value-of-production.zip'
temp_directory= './../temp'
unzip_required_asset(filepath, zip_filepath, temp_directory)

vp = pd.read_csv(filepath, encoding='latin1')
vp = vp.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
vp = vp.loc[vp.Area.isin(shortlist_coutries)]
vp = vp.loc[vp.Item.isin(vp_beef)]
vp = vp.sort_values(by=["Area","Item", "Year"])
vp.reset_index(inplace=True,drop=True)
vp.sample(5)
#SLC standing for Standard Local Currency will require conversion.


# # Trade:

# ## Imports by Value and Quantity & Exports by Value and Quantity

trade_cl_beef = ['Meat, cattle','Butter, cow milk','Milk, whole fresh cow']


filepath = './../temp/trade-crops-livestock.csv'
zip_filepath = './../assets/trade-crops-livestock/'
temp_directory= './../temp'
unzip_required_asset(filepath, zip_filepath, temp_directory)

trade_cl = pd.read_csv(filepath,encoding="latin1")
#trade_cl = trade_cl.loc[trade_cl.Year >= 1990]
trade_cl = trade_cl.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
trade_cl = trade_cl.loc[trade_cl.Area.isin(shortlist_coutries)]
trade_cl = trade_cl.loc[trade_cl.Item.isin(trade_cl_beef)]
trade_cl = trade_cl.sort_values(by=["Area","Item", "Year"])
trade_cl.reset_index(inplace=True,drop=True)
trade_cl.sample(5)


#Splitting datasets into value and wuantity, unity of measure for Quant. in tonnes and for Value uses per 1000 U$
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


#Understand the percentage of missing data per Area
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


#Understand the percentage of missing data per Area
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


population = pd.read_excel("Population Sample Data.xlsx")
population.columns = ["COUNTRY","POPULATION NO.","POP per 1000"]
dic = {}
for i in range(len(population)):
  dic.update({
      population.COUNTRY[i]:population["POPULATION NO."][i]
  })
dic


#Value in dolars
trade_cl_vl_filled["Total_value"] = trade_cl_vl_filled.Value * 1000
trade_cl_vl_filled.tail()

#Add population
pop = []
for i in range(len(trade_cl_vl_filled)):
  pop.append(int(dic[trade_cl_vl_filled.Area[i]].replace(",","")))
trade_cl_vl_filled["Population"] = pop

#Calculate Per Capita
trade_cl_vl_filled["Per Capita"] = trade_cl_vl_filled.Total_value / trade_cl_vl_filled.Population
trade_cl_vl_filled["Per Capita"] = trade_cl_vl_filled["Per Capita"].round(3)


for element in trade_cl_vl_filled.Element.unique():
  for item in trade_cl_vl_filled.Item.unique():
    #Filter by Trade Type and Product Type
    imp = trade_cl_vl_filled.loc[(trade_cl_vl_filled.Element == element)&(trade_cl_vl_filled.Item == item)]
    imp.reset_index(inplace=True,drop=True)
    sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size

    #Plot
    sns.lineplot(x='Year', y='Per Capita',hue="Area",
                data=imp[["Area","Year","Per Capita"]], color="red" ).set_title("{0} -  {1}".format(element,item)) #Plot
    plt.show()


#Value in dolars
trade_cl_qt_filled["Total_value"] = trade_cl_qt_filled.Value * 1000
trade_cl_qt_filled.tail()

#Add population
pop = []
for i in range(len(trade_cl_qt_filled)):
  pop.append(int(dic[trade_cl_qt_filled.Area[i]].replace(",","")))
trade_cl_qt_filled["Population"] = pop

#Calculate Per Capita
trade_cl_qt_filled["Per Capita"] = trade_cl_qt_filled.Total_value / trade_cl_qt_filled.Population
trade_cl_qt_filled["Per Capita"] = trade_cl_qt_filled["Per Capita"].round(3)
trade_cl_qt_filled.head()


for element in trade_cl_qt_filled.Element.unique():
  for item in trade_cl_qt_filled.Item.unique():
    #Filter by Trade Type and Product Type
    imp = trade_cl_qt_filled.loc[(trade_cl_qt_filled.Element == element)&(trade_cl_qt_filled.Item == item)]
    imp.reset_index(inplace=True,drop=True)
    sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size

    #Plot
    sns.lineplot(x='Year', y='Per Capita',hue="Area",
                data=imp[["Area","Year","Per Capita"]], color="red" ).set_title("{0} -  {1}".format(element,item)) #Plot
    plt.show()


# # Supply Utilization Accounts - Crops & Livestock

sua_beef = ['Meat, cattle']

"""sua_beef = ['Beef and Buffalo Meat', 'Beef Mutton Pigmeat Prim''Meat, beef and veal sausages',
       'Meat, beef, dried, salted, smoked', 'Meat, beef, preparations', 'Meat, cattle',
       'Meat, cattle, boneless (beef & veal)', 'Beef and Buffalo Meat', 'Beef Mutton Pigmeat Prim']"""


filepath = './../temp/sua-crops-livestock.csv'
zip_filepath = './../assets/sua-crops-livestock.zip'
temp_directory= './../temp'
unzip_required_asset(filepath, zip_filepath, temp_directory)

sua = pd.read_csv(filepath, encoding='latin1')
#sua = sua.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
sua = sua.loc[sua.Area.isin(shortlist_coutries)]
sua = sua.loc[sua.Item.isin(sua_beef)]
sua = sua.sort_values(by=["Area","Item", "Year"])
sua.reset_index(inplace=True,drop=True)
sua


categories = ['Production', 'Import Quantity', 'Export Quantity']
sua_prod_imp_exp = sua.loc[sua.Element.isin(categories)]
sua_prod_imp_exp.reset_index(inplace=True,drop=True)


sua_prod_imp_exp.Item.unique()


sua_filled_df = pd.DataFrame(columns=sua_prod_imp_exp.columns)

for country in tqdm(list(sua_prod_imp_exp.Area.unique())):
  for type_of_meat in list(sua_prod_imp_exp.Item.unique()):
    for element in list(sua_prod_imp_exp.Element.unique()):
      temp_df = sua_prod_imp_exp.loc[(sua_prod_imp_exp.Area == country) & (sua_prod_imp_exp.Item == type_of_meat ) & (sua_prod_imp_exp.Element == element)]
      temp_df.fillna(temp_df.Value.mean(),inplace=True)
      sua_filled_df = sua_filled_df.append(temp_df)
      sua_filled_df.reset_index(inplace=True,drop=True)

sua_filled_df.dropna(inplace=True)
sua_filled_df.isnull().sum()
sua_filled_df.reset_index(inplace=True,drop=True)


sua_filled_df


sua_filled_df["1000 Tonnes"] = sua_filled_df["Value"] / 1000
sua_filled_df 


"""def create_df(df):
  temp_df = pd.DataFrame()
  temp_df["Production"] = df.loc[df.Element == "Production"]["1000 Tonnes"]
  temp_df["Export Quantity"] = df.loc[df.Element == "Export Quantity"]["1000 Tonnes"]
  temp_df["Import Quantity"] = df.loc[df.Element == "Import Quantity"]["1000 Tonnes"]
  temp_df.insert(0,"Year",list(df.Year.unique()))
"""

def plot(df,region):
  
  df = df.loc[df.Area == region]
  df.reset_index(inplace=True,drop=True)
  sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size
  temp_df = create_df()

  print(temp_df.head())

  sns.lineplot(x='Year', y='value', hue='variable',
             data=pd.melt(temp_df, ['Year'])).set_title("Region: {0} - Trade: {1} (Thousand Tonnes).".format(region))
  plt.show()


"""plot(sua_filled_df,"Argentina")"""


def plot(df,region,meat_type="Meat, Cattle"):
  df = df.loc[df.Area == region]
  df.reset_index(inplace=True,drop=True)
  sns.set(rc = {'figure.figsize':(25,12)}) #Set figure size
  for element in df.Element.unique():
    sns.lineplot(x='Year', y='1000 Tonnes', data=df[["Year","1000 Tonnes"]]).set_title("Region: {0} - Trade: {1} (Thousand Tonnes).".format(region,meat_type)) #Plot
  #plt.savefig("images/"+ region +'.png')
  plt.show()


for region in sua_filled_df.Area.unique():
  plot(sua_filled_df,region)


len(sua_filled_df.Element.unique()) * len(sua_filled_df.Area.unique())


#Distribution of Elements by Year by Region by item


sua.Item.unique()


sua.Element.unique()
#'Production', 'Import Quantity', 'Export Quantity'
#'Food supply (kcal/capita/day)', 'Food supply quantity (g/capita/day)', 'Protein supply quantity (g/capita/day)', 'Fat supply quantity (g/capita/day)'


sua.isnull().sum()


# # Soil Nutrient (STOP)

nutrientflow = ['Cropland nutrient flow']


env_soil_nut = pd.read_csv("Environment_Soil_nutrient_budget.csv", encoding='latin1')
env_soil_nut = env_soil_nut.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
env_soil_nut = env_soil_nut.loc[env_soil_nut.Area.isin(shortlist_coutries)]
env_soil_nut = env_soil_nut.sort_values(by=["Area","Item", "Year"])
env_soil_nut.reset_index(inplace=True,drop=True)

#Unities according to Element
#Cropland nutrient flow per unit area: kg/ha
#Cropland nutrient flow: tonnes


# # Foreign Investiment

fi_item = ['FDI inflows to Agriculture, Forestry and Fishing',
       'Total FDI inflows', 'Total FDI outflows',
       'FDI outflows to Agriculture, Forestry and Fishing']

fi = pd.read_csv('Investment_ForeignDirectInvestment_E_All_Data.csv', encoding='Latin1')
fi = fi.loc[fi.Item.isin(fi_item)]
fi = fi.loc[fi.Area.isin(shortlist_coutries)]
fi = fi.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
fi


# # Government Expenditure

ge_item = ['Total Expenditure (General Government)',
       'Total Expenditure (Central Government)',
       'Agriculture (General Government)',
       'Agriculture (Central Government)']

ge = pd.read_csv('Investment_GovernmentExpenditure_E_All_Data_(Normalized).csv', encoding='latin1')
ge = ge.loc[ge.Area.isin(shortlist_coutries)]
ge = ge.loc[ge.Item.isin(ge_item)]
ge = ge.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
ge.reset_index(inplace=True,drop=True)
ge


# # Country's Investment

country_invest = pd.read_csv('Investment_CountryInvestmentStatisticsProfile_E_All_Data.csv', encoding='latin1')
#country_invest = country_invest.loc[country_invest.Item.isin(prod_item)]
country_invest = country_invest.loc[country_invest.Area.isin(shortlist_coutries)]
country_invest = country_invest.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag', 'Note'], axis=1)
country_invest.reset_index(inplace=True,drop=True)
country_invest


# # Land Use

land_use_item = ['Country area', 'Land area', 'Agriculture']

land_use = pd.read_csv('Inputs_LandUse_E_All_Data.csv', encoding='latin1')
land_use = land_use.loc[land_use.Item.isin(land_use_item)]
land_use = land_use.loc[land_use.Area.isin(shortlist_coutries)]
land_use = land_use.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
land_use.reset_index(inplace=True,drop=True)
land_use


# # Production

prod_item = ['Meat, cattle', 'Milk, whole fresh cow', 'Butter, cow milk']

filepath = './../temp/production-crops-livestock-e-all-data.csv'
zip_filepath = './../assets/production-crops-livestock-e-all-data.zip'
temp_directory= './../temp'
unzip_required_asset(filepath, zip_filepath, temp_directory)

prod = pd.read_csv(filepath, encoding='latin1')
prod = prod.loc[prod.Item.isin(prod_item)]
prod = prod.loc[prod.Area.isin(shortlist_coutries)]
prod = prod.drop(['Area Code', 'Item Code', 'Element Code', 'Year Code', 'Flag'], axis=1)
prod.reset_index(inplace=True,drop=True)
prod


prod_processed = prod.loc[(prod.Element == "Production")&(prod.Year != 2020)]
prod_processed.Element.unique()


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

#Rename Columns
total_df.columns = ["Area","Item","Year","Production_ton","Import_Qt_ton","Export_Qt_ton","Import_Vl_1000$","Export_Vl_1000$"]
total_df


# Mannualy testing previous and new DF to ensure data has remained in place (no shifted rows)

total_df.loc[(total_df.Area == "United States of America") & (total_df.Item == "Butter, cow milk") & (total_df.Year == 2019)]


trade_export_vl.loc[(trade_export_vl.Area == "United States of America") & (trade_export_vl.Item == "Butter, cow milk") & (trade_export_vl.Year == 2019)]


country = total_df.loc[(total_df.Area == "Ireland")&(total_df.Item == "Meat, cattle")&(total_df.Year >=2000)]


plt.figure(figsize=(18,14))
sns.heatmap(country.corr(),annot=True,linecolor="blue",lw=0.5)

