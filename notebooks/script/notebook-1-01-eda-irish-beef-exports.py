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

# ## EDA of the cattle and beef exports (1930 - 2020)

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-1-01-eda-irish-beef-exports.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-1-01-eda-irish-beef-exports.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-1-01-eda-irish-beef-exports.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to provide an Exploratory Data Analysis (EDA) of the `cso-tsa04-exports-of-cattle-and-beef-1930-2020-2022-01Jan-13.csv` file provided by the <a href="https://data.cso.ie/table/TSA04" target="_new">CSO: TSA04 Table</a>. The EDA is performed to investigate and clean the data, to spot anomalies.  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
get_ipython().system('pip install -r script/requirements.txt --quiet --user')
# Remote option
#!pip install -r "https://github.com/markcrowe-com/data-analytics-project-template/blob/master/notebooks/script/requirements.txt?raw=true" --quiet --user


from agriculture_data_analytics.project_manager import ProjectArtifactManager, ProjectAssetManager
from pandas import DataFrame
import data_analytics.exploratory_data_analysis_reports as eda_reports
import data_analytics.github as github
import os
import pandas


artifact_manager:ProjectArtifactManager = ProjectArtifactManager()
asset_manager:ProjectAssetManager = ProjectAssetManager()
artifact_manager.is_remote = asset_manager.is_remote = True
github.display_jupyter_notebook_data_sources(
    [asset_manager.get_cattle_beef_exports_filepath()])
artifact_manager.is_remote = asset_manager.is_remote = False


# ### Working with population estimates CSV file

# #### Create Data Frames

filepath:str = asset_manager.get_cattle_beef_exports_filepath()
beef_export_dataframe:DataFrame = pandas.read_csv(filepath)
beef_export_dataframe.head(5)


# ### Data Type Analysis Quick View

# Print an analysis report of each dataset.  
# - Show the top five rows of the data frame as a quick sample.
# - Show the data types of each column.
# - Report the count of any duplicate rows.
# - Report the counts of any missing values.

beef_export_dataframe.sample(5)


filename:str = os.path.basename(filepath)
eda_reports.print_dataframe_analysis_report(beef_export_dataframe, filename)


# ### Restructure table

beef_export_dataframe = beef_export_dataframe.set_index(
    ['Year', 'Statistic'])['VALUE'].unstack().reset_index()
beef_export_dataframe.columns = beef_export_dataframe.columns.tolist()
beef_export_dataframe[
    "Exports of Cattle"] = beef_export_dataframe["Exports of Cattle"] * 1000
beef_export_dataframe.head()


# #### Renaming Columns

beef_export_dataframe.head(0)


# rename the columns
old_to_new_column_names_dictionary: dict = {
    "Exports of Beef": "Beef Metric Tons",
    "Exports of Cattle": "Cattle"
}
beef_export_dataframe = beef_export_dataframe.rename(
    columns=old_to_new_column_names_dictionary)
beef_export_dataframe.head(0)


eda_reports.print_dataframe_analysis_report(beef_export_dataframe, filename)


# ### Save Artifact

# Saving the output of the notebook.

beef_export_dataframe.to_csv(artifact_manager.get_population_eda_filepath(),
                             index=None)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
