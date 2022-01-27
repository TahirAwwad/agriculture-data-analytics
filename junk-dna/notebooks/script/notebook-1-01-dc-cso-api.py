#!/usr/bin/env python
# coding: utf-8

# ## Download datasets

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-1-01-dc-cso-api.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-1-01-dc-cso-api.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-1-01-dc-cso-api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to download the datasets with the listed in [asset-link-builder.xlsx](./../artifacts/asset-link-builder.xlsx) from [cso.ie](https://wwww.cso.ie).  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from agriculture_data_analytics import cso_ie
from pandas import DataFrame, ExcelFile


# ### Download CSO Data sources

filepath = './../artifacts/asset-link-builder.xlsx'

excelWorkbook: ExcelFile = ExcelFile(filepath)

cso_datasources_dataframe: DataFrame = excelWorkbook.parse('CSO Tables')
excelWorkbook.close()


cso_datasources_dataframe = cso_datasources_dataframe[cso_datasources_dataframe['Download Date'] == '2022-01-19']
cso_datasources_dataframe[['Code', 'Title']]


for _, cso_datasource in cso_datasources_dataframe.iterrows():
    print("Get", cso_datasource['Code'], cso_datasource['Title'])
    dataframe = cso_ie.download_cso_table_dataframe(cso_datasource['Code'])
    filepath: str = f"./../assets/{cso_datasource['Filename']}"
    dataframe.to_csv(filepath, index=False)
    print(dataframe.head())
    print(f'Saved to "{filepath}"')
    print()


# ### Normalize CSO Data sources

# #### AEA01 Value at Current Prices for Output, Input and Income in Agriculture

import pandas
filepath: str = './../assets/cso-aea01-value-at-current-prices-for-output-input-and-income-in-agriculture.csv'
prc_9021df = pandas.read_csv(filepath).pivot_table(columns="Statistic",
                                                   index=['Year', 'UNIT'],
                                                   values='VALUE',
                                                   dropna=True).reset_index()

prc_9021df.to_csv('./../artifacts/TA_inputoutputvalue_1990_2021_CSO.csv')


# #### AEA05 Value at Current Prices for Subsidies on Products

filepath: str = './../assets/cso-aea05-value-at-current-prices-for-subsidies-on-products.csv'

subsidies_df = pandas.read_csv(filepath).pivot_table(
    columns="Statistic", index=['Year', 'UNIT'], values='VALUE',
    dropna=True).reset_index()

#subsidies_df.to_csv('./../artifacts/TA_subsidies_1990_2020_CSO.csv')


# #### AHA01 Agricultural Input and Output Price Indices

filepath: str = './../assets/cso-aha01-agricultural-input-and-output-price-indices.csv'

prc_idx_9510_df = pandas.read_csv(filepath).pivot_table(
    columns="Agricultural Product",
    index=['Year', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

#prc_idx_9510_df.to_csv(    './../artifacts/TA_inputoutputpriceIDX_1995_2010_CSO.csv')


# #### AHA03 Agricultural Input and Output Price Indices

filepath: str = './../assets/cso-aha03-agricultural-input-and-output-price-indices.csv'

prc_idx_0517_df = pandas.read_csv(filepath).pivot_table(
    columns="Agricultural Product",
    index=['Year', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

#prc_idx_0517_df.to_csv('./../artifacts/TA_inputoutputpriceIDX_2005_2017_CSO.csv')


# #### AHA04 Agricultural Input and Output Price Indices (Base 2015=100)

filepath: str = './../assets/cso-aha04-agricultural-input-and-output-price-indices.csv'

prc_idx_1420_df = pandas.read_csv(filepath).pivot_table(
    columns="Agricultural Product",
    index=['Year', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

#prc_idx_1420_df.to_csv('./../artifacts/TA_inputoutputpriceIDX_2014_2020_CSO.csv')


# #### AQA03 Crop Yield 1985-2007

filepath: str = './../assets/cso-aqa03-crop-yield-1985-2007.csv'

crop_yield8507_df = pandas.read_csv(filepath).pivot_table(
    columns="Statistic",
    index=['Year', 'Type of Crop', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

#crop_yield8507_df.to_csv('./../artifacts/TA_cropyield_1985_2007_CSO.csv')


# #### AQA04 Crop Yield and Production

filepath: str = './../assets/cso-aqa04-crop-yield-and-production.csv'

crop_yield0820_df = pandas.read_csv(filepath).pivot_table(
    columns="Statistic",
    index=['Year', 'Type of Crop', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index().rename(
        columns={"Crop Production": "Crop Yield"})

#crop_yield0820_df.to_csv('./../artifacts/TA_cropyield_2008_2020_CSO.csv')


# Join Crop Yields from 1985 to 2020 into 1 dataframe

# append crop yield from 1985 tp 2020
crop_yield_ie_df = crop_yield8507_df.append(crop_yield0820_df)

#crop_yield_ie_df.to_csv('./../artifacts/TA_cropyield_1985_2020_CSO.csv')

