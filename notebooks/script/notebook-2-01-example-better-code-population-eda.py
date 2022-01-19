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

# ## EDA of the population (1950 - 2021)

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "data-analytics-project-template", "notebooks/notebook-2-01-example-better-code-population-eda.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/data-analytics-project-template/master?filepath=notebooks/notebook-2-01-example-better-code-population-eda.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/data-analytics-project-template/blob/master/notebooks/notebook-2-01-example-better-code-population-eda.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to provide an Exploratory Data Analysis (EDA) of the `2021-12Dec-11-population-estimates-1950-2021-pea01.csv` file provided by the <a href="https://data.cso.ie/table/PEA01" target="_new">CSO: PEA01 Table</a>. The EDA is performed to investigate and clean the data, to spot anomalies.  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt --quiet
# Remote option
#!pip install -r https://github.com/markcrowe-com/data-analytics-project-template/blob/master/notebooks/script/requirements.txt --quiet


from population_planning.dataframe_labels import *
from population_planning.project_manager import ProjectArtifactManager, ProjectAssetManager
import data_analytics.github as github
import data_analytics.exploratory_data_analysis_reports as eda_reports
import os
import pandas


artifact_manager: ProjectArtifactManager = ProjectArtifactManager()
asset_manager: ProjectAssetManager = ProjectAssetManager()
artifact_manager.is_remote = asset_manager.is_remote = True
github.display_jupyter_notebook_data_sources(
    [asset_manager.get_population_estimates_filepath()])
artifact_manager.is_remote = asset_manager.is_remote = False


# ### Working with population estimates CSV file

# #### Create Data Frames

filepath: str = asset_manager.get_population_estimates_filepath()
population_dataframe = pandas.read_csv(filepath)
population_dataframe.head(0)


# #### Renaming Columns

# rename the columns
old_to_new_column_names_dictionary = {"Age Group": AGE, "VALUE": POPULATION}
population_dataframe = population_dataframe.rename(
    columns=old_to_new_column_names_dictionary)
population_dataframe.head(0)


# ### Data Type Analysis Quick View

# Print an analysis report of each dataset.  
# - Show the top five rows of the data frame as a quick sample.
# - Show the data types of each column.
# - Report the count of any duplicate rows.
# - Report the counts of any missing values.

filename = os.path.basename(filepath)
eda_reports.print_dataframe_analysis_report(population_dataframe, filename)


population_dataframe = population_dataframe.drop(["Statistic"], axis="columns")
population_dataframe = population_dataframe.drop(["UNIT"], axis="columns")


# ### Remove duplicate Ranges

duplicate_age_range_labels = [
    "0 - 4 years", "0 - 14 years", "15 - 24 years", "15 years and over",
    "25 - 44 years", "45 - 64 years", "65 years and over", "All ages"
]

population_dataframe = population_dataframe.drop(population_dataframe[(
    population_dataframe[AGE].isin(duplicate_age_range_labels))].index)
print(population_dataframe.isnull().sum())


# 'Both sexes' - is not needed for the population pyramids

population_dataframe = population_dataframe.drop(
    population_dataframe[(population_dataframe[SEX] == "Both sexes")].index)
population_dataframe.where(population_dataframe[SEX] == "Both sexes").count(
    axis=0)


# ### Rename age brackets for chart labels

population_dataframe[AGE].unique()


def rename_age_bracket(label: str) -> str:
    if label == "Under 1 year":
        return "00"
    if label == "85 years and over":
        return "85+"
    label = label.replace(" years", "")
    if label == "1 - 4" or label == "5 - 9":
        return "0" + label.replace(" - ", "-0")
    return label.replace(" ", "")


for value in population_dataframe[AGE].unique():
    population_dataframe[AGE] = population_dataframe[AGE].replace(
        value, rename_age_bracket(value))

population_dataframe.head()


# ### Save Artifact
# Saving the output of the notebook.

population_dataframe.to_csv(artifact_manager.get_population_eda_filepath(),
                            index=None)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
