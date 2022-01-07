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
import pandas
import numpy
import spacy


#!python -m spacy download en_core_web_sm --quiet


def preprocess(html_text):
    return html_text.str.replace("(<br/>)", "", regex=True
                        ).replace('(<a).*(>).*(</a>)', '', regex=True
                        ).replace('(&amp)', '', regex=True
                        ).replace('(&gt)', '', regex=True
                        ).replace('(&lt)', '',regex=True
                        ).replace('(\xa0)', ' ',regex=True)

def spacy_clean(panda):  
        delete=['NUM','INTJ','CONJ','ADV','PUNCT','PART','DET','ADP','SPACE','PRON','SYM','x']
        list = []
        doc= nlp(panda)
        for token in doc:
            if token.is_stop == False and token.is_alpha and len(token)>2 and token.pos_ not in delete:
                group = token.lemma_
                list.append(group)
        return list


dataframe = pandas.read_csv("./../assets/agriculture-ie.csv")

nlp = spacy.load('en_core_web_sm')


#Text preprocessing
dataframe["text"] = dataframe["text"].str.lower()
dataframe['text'] = preprocess(dataframe['text'])
#Lower Casing
dataframe["text"] = dataframe["text"].str.lower()

##loading spacy english language for spacy
## TODO: ERROR
#nlp = spacy.load('en', parser=False, entity=False)

#spacy function that removes unwanted words from Twitter posts

dataframe['clean_text'] = dataframe['text'].apply(lambda x:spacy_clean(x))
dataframe['clean_text'] = [' '.join(map(str, l)) for l in dataframe['text']]

dataframe = dataframe.dropna()


#This needs fixing
#changing tokens into strings
for i, text in enumerate(dataframe["text"]):
    tokens = text.split(" ")
    new_text = []
    for t in tokens:
        #if t in words:
        new_text.append(t)

    new_text = " ".join(new_text)
    dataframe["clean_text"][i] = new_text

#obtaining polarity and word count

dataframe['polarity_tokens'] = dataframe['clean_text'].map(lambda text: TextBlob(text).sentiment.polarity)
dataframe['review_len'] = dataframe['clean_text'].astype(str).apply(len)
dataframe['word_count'] = dataframe['clean_text'].apply(lambda x: len(str(x).split()))


# ### Save Artifact
# Saving the output of the notebook.

dataframe.to_csv("./../artifacts/agriculture-ie-clean.csv")

