#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from textblob import TextBlob

import spacy
from pprint import pprint

#nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords

#from spellchecker import SpellChecke


#loading of the csv
df = pd.read_csv("./../artifacts/agriculture_ie.csv")

print('test')                 


# In[152]:
nlp = spacy.load('en')

#Text preprocessing 

df["text"] = df["text"].str.lower()

def preprocess(CleanText):
    CleanText = CleanText.str.replace("(<br/>)", "", regex=True)
    CleanText = CleanText.str.replace('(<a).*(>).*(</a>)', '', regex=True)
    CleanText = CleanText.str.replace('(&amp)', '', regex=True)
    CleanText = CleanText.str.replace('(&gt)', '', regex=True)
    CleanText = CleanText.str.replace('(&lt)', '',regex=True)
    CleanText = CleanText.str.replace('(\xa0)', ' ',regex=True)  
    return CleanText

df['text'] = preprocess(df['text'])


#Lower Casing

df["text"] = df["text"].str.lower()


##loading spacy english language for spacy
nlp = spacy.load('en', parser=False, entity=False)

#spacy function that removes unwanted words from Twitter posts

def spacy_clean(panda):  
        delete=['NUM','INTJ','CONJ','ADV','PUNCT','PART','DET','ADP','SPACE','PRON','SYM','x']
        list = []
        doc= nlp(panda)
        for token in doc:
            if token.is_stop == False and token.is_alpha and len(token)>2 and token.pos_ not in delete:
                group = token.lemma_
                list.append(group)
        return list

df['clean_text']=df['text'].apply(lambda x:spacy_clean(x))

df['clean_text'] = [' '.join(map(str, l)) for l in df['text']]



df = df.dropna()

#changing tokens into strings

for i, text in enumerate(df["text"]):
    tokens = text.split(" ")
    new_text = []
    for t in tokens:
        if t in words:
            new_text.append(t)

    new_text = " ".join(new_text)
    df["clean_text"][i] = new_text

#obtaining polarity and word count

df['polarity_tokens'] = df['clean_text'].map(lambda text: TextBlob(text).sentiment.polarity)
df['review_len'] = df['clean_text'].astype(str).apply(len)
df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
print('t')

#processed file save
df.to_csv("C:/Users/Swazy/agriculture_ie_clean.csv")





conda install -c conda-forge spacy-model-en_core_web_lg




