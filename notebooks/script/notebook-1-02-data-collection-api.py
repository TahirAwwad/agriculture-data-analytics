#!/usr/bin/env python
# coding: utf-8

# ## Fetch the proposed datasets understudy

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-1-01-api-data-collection.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-1-01-api-data-collection.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-1-01-api-data-collection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt --quiet
# Remote option
#!pip install -r https://github.com/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/script/requirements.txt --quiet


from pandas import DataFrame
import eurostat
import io
import pandas
import requests


def download_cso_table_data(table_code: str,
                            file_type: str = "CSV",
                            version: str = "1.0") -> str:
    BASE_URL = "https://ws.cso.ie/public/api.jsonrpc"
    JSON_DATA = f'{{"jsonrpc":"2.0","method":"PxStat.Data.Cube_API.ReadDataset","params":{{"class":"query","id":[],"dimension":{{}},"extension":{{"pivot":null,"codes":false,"language":{{"code":"en"}},"format":{{"type":"{file_type}","version":"{version}"}},"matrix":"{table_code}"}},"version":"2.0"}}}}'

    url = f"{BASE_URL}?data={JSON_DATA}"
    response_json_rpc = requests.get(url).json()
    print(f"Downloaded https://data.cso.ie/table/{table_code}")
    return response_json_rpc['result']


def download_cso_table_dataframe(table_code: str) -> str:
    return pandas.read_csv(io.StringIO(download_cso_table_data(table_code)),
                           sep=",")


# ### Source 1: Eurostat data

# - Agriculture price indecies of product

# #### 2015 base year of Product Prices

# price indecies of product 2015 base year
price_idx_products_annual_code = 'apri_pi15_outa'

price_idx_products_annual_dataframe = eurostat.get_data_df(
    price_idx_products_annual_code, flags=False)

# rename column
price_idx_products_annual2015_dataframe = price_idx_products_annual_dataframe.rename(
    columns={price_idx_products_annual_dataframe.columns[3]: 'geotime'})

# transform years columns to a Series
price_idx_products_annual2015_dataframe = price_idx_products_annual2015_dataframe.melt(
    id_vars=["p_adj", "unit", "geotime", "product"],
    var_name="year",
    value_name="priceIDX")
price_idx_products_annual2015_dataframe.sample(5)


# #### 2010 base year of Product Prices

# price indecies of product 2010 base year
price_idx_products_annual2010_code = 'apri_pi10_outa'
price_idx_products_annual2010_dataframe = eurostat.get_data_df(
    price_idx_products_annual2010_code, flags=False)

price_idx_products_annual2010_dataframe = price_idx_products_annual2010_dataframe.rename(
    columns={price_idx_products_annual2010_dataframe.columns[3]: 'geotime'})

# transform years columns to a Series
price_idx_products_annual2010_dataframe = price_idx_products_annual2010_dataframe.melt(
    id_vars=["p_adj", "unit", "geotime", "product"],
    var_name="year",
    value_name="priceIDX")

price_idx_products_annual2010_dataframe.to_csv(
    './../assets/TA_priceIDX_2000_2017_eurostat.csv')


# ### Download CSO Data sources

filepath = './../artifacts/asset-link-builder.xlsx'

excelWorkbook = pandas.ExcelFile(filepath)

cso_datasources_dataframe: DataFrame = excelWorkbook.parse('CSO Tables')
excelWorkbook.close()


cso_datasources_dataframe = cso_datasources_dataframe[cso_datasources_dataframe['Download Date'] == '2022-01-19']
cso_datasources_dataframe[['Code', 'Title']]


for _, cso_datasource in cso_datasources_dataframe.iterrows():
    print("Get", cso_datasource['Code'], cso_datasource['Title'])
    dataframe = download_cso_table_dataframe(cso_datasource['Code'])
    filepath: str = f"./../assets/{cso_datasource['Filename']}"
    dataframe.to_csv(filepath)
    print(filepath)


# ### Normalize CSO Data sources

# #### AEA01 Value at Current Prices for Output, Input and Income in Agriculture

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

subsidies_df.to_csv('./../artifacts/TA_subsidies_1990_2020_CSO.csv')


# #### AHA01 Agricultural Input and Output Price Indices

filepath: str = './../assets/cso-aha01-agricultural-input-and-output-price-indices.csv'

prc_idx_9510_df = pandas.read_csv(filepath).pivot_table(
    columns="Agricultural Product",
    index=['Year', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

prc_idx_9510_df.to_csv(
    './../artifacts/TA_inputoutputpriceIDX_1995_2010_CSO.csv')


# #### AHA03 Agricultural Input and Output Price Indices

filepath: str = './../assets/cso-aha03-agricultural-input-and-output-price-indices.csv'

prc_idx_0517_df = pandas.read_csv(filepath).pivot_table(
    columns="Agricultural Product",
    index=['Year', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

prc_idx_0517_df.to_csv(
    './../artifacts/TA_inputoutputpriceIDX_2005_2017_CSO.csv')


# #### AHA04 Agricultural Input and Output Price Indices (Base 2015=100)

filepath: str = './../assets/cso-aha04-agricultural-input-and-output-price-indices.csv'

prc_idx_1420_df = pandas.read_csv(filepath).pivot_table(
    columns="Agricultural Product",
    index=['Year', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

prc_idx_1420_df.to_csv(
    './../artifacts/TA_inputoutputpriceIDX_2014_2020_CSO.csv')


# #### AQA03 Crop Yield 1985-2007

filepath: str = './../assets/cso-aqa03-crop-yield-1985-2007.csv'

crop_yield8507_df = pandas.read_csv(filepath).pivot_table(
    columns="Statistic",
    index=['Year', 'Type of Crop', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index()

crop_yield8507_df.to_csv('./../artifacts/TA_cropyield_1985_2007_CSO.csv')


# #### AQA04 Crop Yield and Production

filepath: str = './../assets/cso-aqa04-crop-yield-and-production.csv'

crop_yield0820_df = pandas.read_csv(filepath).pivot_table(
    columns="Statistic",
    index=['Year', 'Type of Crop', 'UNIT'],
    values='VALUE',
    dropna=True).reset_index().rename(
        columns={"Crop Production": "Crop Yield"})

crop_yield0820_df.to_csv('./../artifacts/TA_cropyield_2008_2020_CSO.csv')


# Join Crop Yields from 1985 to 2020 into 1 dataframe

# append crop yield from 1985 tp 2020
crop_yield_ie_df = crop_yield8507_df.append(crop_yield0820_df)

crop_yield_ie_df.to_csv('./../artifacts/TA_cropyield_1985_2020_CSO.csv')

