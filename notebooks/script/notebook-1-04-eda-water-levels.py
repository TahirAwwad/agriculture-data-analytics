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

# ## EDA of Water Levels

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "data-analytics-project-template",
#                                             "notebooks/notebook-1-04-eda-water-levels.ipynb"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/data-analytics-project-template/master?filepath=notebooks/notebook-1-04-eda-water-levels.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/data-analytics-project-template/blob/master/notebooks/notebook-1-04-eda-water-levels.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective
# The objective is to provide an Exploratory Data Analysis (EDA) of the water levels data files. The EDA is performed to investigate and clean the data, to spot anomalies.  
# ### Setup
# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt --quiet
# Remote option
#!pip install -r https://github.com/tahirawwad/agriculture-data-analytics/blob/master/notebooks/script/requirements.txt --quiet


from population_planning.dataframe_labels import *
from population_planning.project_manager import ProjectArtifactManager, ProjectAssetManager
import data_analytics.github as github
import data_analytics.exploratory_data_analysis_reports as eda_reports
import os
import pandas


artifact_manager = ProjectArtifactManager()
asset_manager = ProjectAssetManager()
artifact_manager.is_remote = asset_manager.is_remote = True
github.display_jupyter_notebook_data_sources([asset_manager.get_population_estimates_filepath()])
artifact_manager.is_remote = asset_manager.is_remote = False


# ### Working with population estimates CSV file
# #### Create Data Frames

filepath = "./../assets/epa-groundwater-monitoring-data-to-end-2020-circulation-26.05.21.xlsx"

excelWorkbook = pandas.ExcelFile(filepath)
worksheet_name = 'Data'
ground_water_dataframe = excelWorkbook.parse(worksheet_name);


import re as RegularExpression
def camel_to_snakecase(name: str) -> str:
    """
    Convert CamelCase to snake_case
    :param name: string
    :return: snake_case string
    """
    name = RegularExpression.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return RegularExpression.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def english_to_snakecase(name: str) -> str:
    """
    Convert CamelCase to snake_case
    :param name: string
    :return: snake_case string
    """
    return name.lower().replace(" ", "-")

english_to_snakecase("EPA Groundwater Monitoring Data to End 2020 Circulation 26.05.21")


# #### Renaming Columns

# rename the columns
old_to_new_column_names_dictionary = {
                                      "Age Group" : AGE,
                                      "VALUE" : POPULATION
                                     }
#ground_water_dataframe = ground_water_dataframe.rename(columns = old_to_new_column_names_dictionary)
#ground_water_dataframe.head(0)


# ### Data Type Analysis Quick View
# Print an analysis report of each dataset.  
# - Show the top five rows of the data frame as a quick sample.
# - Show the data types of each column.
# - Report the count of any duplicate rows.
# - Report the counts of any missing values.

filename = os.path.basename(filepath)
eda_reports.print_dataframe_analysis_report(ground_water_dataframe, filename)


# ### Save Artifact
# Saving the output of the notebook.

#population_dataframe.to_csv("./../artifacts/water-levels-cleaned.csv", index=None)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
