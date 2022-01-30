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

# ## EDA of Irish Bovine Tuberculosis

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-2-02-eda-irish-bovine-tuberculosis.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-2-02-eda-irish-bovine-tuberculosis.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-2-02-eda-irish-bovine-tuberculosis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to provide an Exploratory Data Analysis (EDA) of the `cso-daa01-bovine-tuberculosis-2022-01-Jan-15.csv` file provided by the <a href="https://data.cso.ie/table/DAA01" target="_new">CSO: DAA01 Table</a>. The EDA is performed to investigate and clean the data, to spot anomalies.  

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

filepath: str = "./../assets/cso-daa01-bovine-tuberculosis.csv"
bovine_tuberculosis_dataframe: DataFrame = pandas.read_csv(filepath)


# #### Renaming Columns

old_to_new_column_names_dictionary = {
    "Regional Veterinary Offices": VETERINARY_OFFICE,
    "Herds in County": HERDS_COUNT,
    "Animals in County": ANIMAL_COUNT,
    "Herds Restricted since 1st of January": RESTRICTED_HERDS_AT_START_OF_YEAR,
    "Herds Restricted by 31st of December": RESTRICTED_HERDS_AT_END_OF_YEAR,
    "Herds Tested": HERDS_TESTED,
    "Herd Incidence": HERD_INCIDENCE_RATE,
    "Tests on Animals": TESTS_ON_ANIMALS,
    "Reactors per 1000 Tests A.P.T.": REACTORS_PER_1000_TESTS_APT,
    "Reactors to date": REACTORS_TO_DATE,
    UNIT.upper(): UNIT,
    VALUE.upper(): VALUE,
}
bovine_tuberculosis_dataframe = bovine_tuberculosis_dataframe.rename(
    columns=old_to_new_column_names_dictionary)
bovine_tuberculosis_dataframe.head(0)


# #### Data Type Analysis Quick View

# Print an analysis report of each dataset.  
# - Show the top five rows of the data frame as a quick sample.
# - Show the data types of each column.
# - Report the count of any duplicate rows.
# - Report the counts of any missing values.

filename: str = os.path.basename(filepath)
eda_reports.print_dataframe_analysis_report(bovine_tuberculosis_dataframe, filename)


# ### Normalizing the table

bovine_tuberculosis_dataframe = bovine_tuberculosis_dataframe.set_index(
    [YEAR, VETERINARY_OFFICE, STATISTIC])[VALUE].unstack().reset_index()
bovine_tuberculosis_dataframe.columns = bovine_tuberculosis_dataframe.columns.tolist()
bovine_tuberculosis_dataframe = bovine_tuberculosis_dataframe.rename(
    columns=old_to_new_column_names_dictionary)


bovine_tuberculosis_dataframe.head()


# #### Data Type Analysis Quick View

eda_reports.print_dataframe_analysis_report(bovine_tuberculosis_dataframe, filename)


# The table contains both data for county level and state level an aggregate of the county level data

county_bovine_tuberculosis_dataframe = bovine_tuberculosis_dataframe.drop(bovine_tuberculosis_dataframe[(bovine_tuberculosis_dataframe[VETERINARY_OFFICE] == "State")].index)
eda_reports.print_dataframe_analysis_report(county_bovine_tuberculosis_dataframe, filename)


bovine_tuberculosis_dataframe = bovine_tuberculosis_dataframe.drop(bovine_tuberculosis_dataframe[(bovine_tuberculosis_dataframe[VETERINARY_OFFICE] != "State")].index)
eda_reports.print_dataframe_analysis_report(bovine_tuberculosis_dataframe, filename)


# ### Save Artifacts

# Saving the output of the notebook.

bovine_tuberculosis_dataframe.to_csv(
    artifact_manager.get_bovine_tuberculosis_eda_filepath(), index=None)
county_bovine_tuberculosis_dataframe.to_csv(
    artifact_manager.get_county_bovine_tuberculosis_eda_filepath(), index=None)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
