#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-3-02-clean-farm.ipynb", "master")
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-3-02-clean-farm.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-3-02-clean-farm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Setup
# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt --quiet
# Remote option
#!pip install -r https://github.com/tahirawwad/agriculture-data-analytics/blob/master/notebooks/script/requirements.txt --quiet


from nltk.corpus import stopwords
from pprint import pprint
from textblob import TextBlob
import importlib
import pandas
import numpy
import spacy


nlp_functions = importlib.import_module("nlp_clean_functions")


#!python -m spacy download en_core_web_sm --quiet


get_ipython().run_cell_magic('time', '', 'dataframe = pandas.read_csv("./../assets/beef.csv")\ndataframe = nlp_functions.add_clean_text_columns(dataframe)')


# ### Save Artifact
# Saving the output of the notebook.

dataframe.to_csv("./../artifacts/beef-clean.csv")


dataframe.head()




