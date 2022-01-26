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
#                                             "junk-dna/notebooks/notebook-1-04-eda-water-levels.ipynb"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/data-analytics-project-template/master?filepath=junk-dna/notebooks/notebook-1-04-eda-water-levels.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/data-analytics-project-template/blob/master/junk-dna/notebooks/notebook-1-04-eda-water-levels.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to provide an Exploratory Data Analysis (EDA) of the water levels data files. The EDA is performed to investigate and clean the data, to spot anomalies.  Data sourced from [EPA: Water Quality and Monitoring, Groundwater Quality (Excel) 1990 - 2020](https://gis.epa.ie/GetData/Download)

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt --quiet
# Remote option
#!pip install -r https://github.com/tahirawwad/agriculture-data-analytics/blob/master/junk-dna/notebooks/script/requirements.txt --quiet


from pandas import DataFrame
import data_analytics.exploratory_data_analysis as eda
import data_analytics.exploratory_data_analysis_reports as eda_reports
import matplotlib.pyplot as pyplot
import numpy
import os
import pandas
import seaborn


# ### Working with population estimates CSV file

# #### Create Data Frames

filepath = "./../assets/3_months.csv"

water_level_dataframe = pandas.read_csv(filepath, delimiter=";", skiprows=7)


water_level_dataframe.head()


filepath = "./../assets/epa-groundwater-monitoring-data-to-end-2020-circulation-26.05.21.xlsx"

excelWorkbook = pandas.ExcelFile(filepath)
worksheet_name = 'Data'
ground_water_dataframe = excelWorkbook.parse(worksheet_name)
excelWorkbook.close()  # Close Connection to the open excelWorkbook


eda_reports.print_dataframe_analysis_report(water_level_dataframe)


# ### Data Type Analysis Quick View

# Print an analysis report of each dataset.  
# - Show the top five rows of the data frame as a quick sample.
# - Show the data types of each column.
# - Report the count of any duplicate rows.
# - Report the counts of any missing values.

missing_value_stats_dataframe = eda.calculate_missing_value_statistics(ground_water_dataframe)

missing_value_stats_dataframe


# A function to visualize missing data in a DataFrame

def barplot_missing_values(missing_value_stats_dataframe: DataFrame) :
    pyplot.subplots(figsize=(40, 30))
    seaborn.barplot(x=missing_value_stats_dataframe.index, y='% Missing', data=missing_value_stats_dataframe)
    pyplot.xticks(rotation=90)
    pyplot.show()


barplot_missing_values(missing_value_stats_dataframe)


columns: list = []
for index, row in missing_value_stats_dataframe.iterrows():
    if row["% Missing"] > .8:
        columns.append(index)
ground_water_dataframe.drop(columns, axis=1, inplace=True)


dataframe_columns = ground_water_dataframe.columns.values.tolist()

print("Columns:", len(dataframe_columns))

for column in dataframe_columns[3:-1]:
    try:
        ground_water_dataframe[column] = ground_water_dataframe[column].astype(
            'int32')
    except:
        print(ground_water_dataframe[column].unique())


ground_water_dataframe.replace("-", numpy.nan, inplace=True)


filename = os.path.basename(filepath)

ground_water_dataframe[["Easting", "Northing"]] = ground_water_dataframe[[
    "Easting", "Northing"
]].apply(pandas.to_numeric)


ground_water_dataframe.replace("--", numpy.nan, inplace=True)


ground_water_dataframe[["Temperature (on-site)"]] = ground_water_dataframe[[
    "Temperature (on-site)"
]].apply(pandas.to_numeric)


missing_value_stats_dataframe = eda.calculate_missing_value_statistics(ground_water_dataframe)
barplot_missing_values(missing_value_stats_dataframe)


eda_reports.print_dataframe_analysis_report(ground_water_dataframe, filename)


# ### Save Artifact

# Saving the output of the notebook.

#population_dataframe.to_csv("./../artifacts/water-levels-cleaned.csv", index=None)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
