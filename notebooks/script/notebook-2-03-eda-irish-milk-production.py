#!/usr/bin/env python
# coding: utf-8

# <style>
# *
# {
# 	text-align: justify;
# 	line-height: 1.5;
# 	font-family: "Arial", sans-serif;
# 	font-size: 12px;
# }
# 
# h2, h3, h4, h5, h6
# {
# 	font-family: "Arial", sans-serif;
# 	font-size: 12px;
# 	font-weight: bold;
# }
# h2
# {
# 	font-size: 14px;
# }
# h1
# {
# 	font-family: "Wingdings", sans-serif;
# 	font-size: 16px;
# }
# </style>

# ## EDA of Irish agriculture output, input, income values at current prices.

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-2-03-eda-irish-milk-production.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-2-03-eda-irish-milk-production.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-2-03-eda-irish-milk-production.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to provide an Exploratory Data Analysis (EDA) of the [cso-aea01-value-at-current-prices-for-output-input-and-income-in-agriculture.csv](./../assets/cso-aea01-value-at-current-prices-for-output-input-and-income-in-agriculture.csv) file provided by the <a href="https://data.cso.ie/table/AEA01" target="_new">CSO: AEA01 Table</a>. The EDA is performed to investigate and clean the data, to spot anomalies.  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from agriculture_data_analytics.project_manager import *
from agriculture_data_analytics.dataframe_labels import *
from pandas import DataFrame
import data_analytics.exploratory_data_analysis as eda
import data_analytics.exploratory_data_analysis_reports as eda_reports
import data_analytics.github as github
import os
import pandas


artifact_manager: ProjectArtifactManager = ProjectArtifactManager()
asset_manager: ProjectAssetManager = ProjectAssetManager()
artifact_manager.is_remote = asset_manager.is_remote = True
github.display_jupyter_notebook_data_sources(
    [asset_manager.get_bovine_tuberculosis_filepath()])
artifact_manager.is_remote = asset_manager.is_remote = False


# ### Loading the CSV file

# #### Create Data Frames

filepath: str = './../assets/cso-aea01-value-at-current-prices-for-output-input-and-income-in-agriculture.csv'
agriculture_prices_dataframe: DataFrame = pandas.read_csv(filepath)


# #### Renaming Columns

old_to_new_column_names_dictionary = {
    UNIT.upper(): UNIT,
    VALUE.upper(): VALUE,
}
agriculture_prices_dataframe = agriculture_prices_dataframe.rename(
    columns=old_to_new_column_names_dictionary)
agriculture_prices_dataframe.head(0)


# #### Data Type Analysis Quick View

# Print an analysis report of each dataset.  
# - Show the top five rows of the data frame as a quick sample.
# - Show the data types of each column.
# - Report the count of any duplicate rows.
# - Report the counts of any missing values.

filename: str = os.path.basename(filepath)
eda_reports.print_dataframe_analysis_report(agriculture_prices_dataframe, filename)


agriculture_prices_dataframe.drop([STATE, UNIT], axis=1, inplace=True)


agriculture_prices_dataframe.sample()


# ### Normalizing the table

agriculture_prices_dataframe = agriculture_prices_dataframe.pivot_table(
    columns=STATISTIC, index=[YEAR], values=VALUE,
    dropna=True).reset_index().rename_axis(None, axis=1)


agriculture_prices_dataframe.sample()


# #### Data Type Analysis Quick View

eda_reports.print_dataframe_analysis_report(agriculture_prices_dataframe, filename)


# #### Examine the null values

eda_reports.print_columns_rows_with_missing_values(agriculture_prices_dataframe, [YEAR])


# The record for the year 2021 seems to be largely incomplete. 

agriculture_prices1_dataframe = agriculture_prices_dataframe[(agriculture_prices_dataframe[YEAR] != 2021)]


eda_reports.report_missing_values(agriculture_prices1_dataframe)


eda_reports.print_columns_rows_with_missing_values(agriculture_prices1_dataframe, [YEAR])


# #### Data Type Analysis Quick View

eda_reports.print_dataframe_analysis_report(agriculture_prices1_dataframe, filename)


# ### Save Artifact

# Saving the output of the notebook.

agriculture_prices1_dataframe.to_csv(
    './../artifacts/irish-milk-production-eda-output.csv', index=None)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
