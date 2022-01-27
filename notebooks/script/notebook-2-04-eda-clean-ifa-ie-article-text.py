#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-3-02-clean-farm.ipynb", "master")
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-3-02-clean-farm.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-3-02-clean-farm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# Extract and clean text

# ### Setup
# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from nltk.corpus import stopwords
from pprint import pprint
from textblob import TextBlob
from agriculture_data_analytics import sentiment_analysis
import importlib
import pandas
import numpy
import spacy


#!python -m spacy download en_core_web_sm --quiet


# ### Loading the CSV file

filename: str = "./../assets/ifa-ie-articles.csv"
dataframe = pandas.read_csv(filename)


# ### Clean Text

dataframe = sentiment_analysis.add_clean_text_columns(dataframe, 'Text')


# #### Examine results

dataframe.head()


dataframe['clean_text'].iloc[0]


# ### Save Artifact
# Saving the output of the notebook.

filename: str = "./../artifacts/ifa-ie-articles.csv"
dataframe.to_csv(filename, index=False)

