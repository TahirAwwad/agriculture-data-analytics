#!/usr/bin/env python
# coding: utf-8

# ## Download eurostat datasets

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-1-02-dc-eurostat-api.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-1-02-dc-eurostat-api.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-1-02-dc-eurostat-api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to download the datasets from the [eurostat](https://ec.europa.eu/eurostat) API.  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from pandas import DataFrame
import eurostat


# ### Download Eurostat Data sources

# - Agriculture price indecies of product

# #### 2015 base year of Product Prices

# price indecies of product 2015 base year
price_idx_products_annual_code: str = 'apri_pi15_outa'

price_idx_products_annual_dataframe: DataFrame = eurostat.get_data_df(
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
price_idx_products_annual2010_code: str = 'apri_pi10_outa'
price_idx_products_annual2010_dataframe: DataFrame = eurostat.get_data_df(
    price_idx_products_annual2010_code, flags=False)

price_idx_products_annual2010_dataframe = price_idx_products_annual2010_dataframe.rename(
    columns={price_idx_products_annual2010_dataframe.columns[3]: 'geotime'})

# transform years columns to a Series
price_idx_products_annual2010_dataframe = price_idx_products_annual2010_dataframe.melt(
    id_vars=["p_adj", "unit", "geotime", "product"],
    var_name="year",
    value_name="priceIDX")

filepath:str ='./../assets/TA_priceIDX_2000_2017_eurostat.csv'
#price_idx_products_annual2010_dataframe.to_csv(filepath, index=False)

