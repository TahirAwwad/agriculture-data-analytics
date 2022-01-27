#!/usr/bin/env python
# coding: utf-8

# ## Download CSO datasets

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

