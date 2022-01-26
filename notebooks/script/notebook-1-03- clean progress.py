#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from textblob import TextBlob

import spacy
from pprint import pprint

nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords

#from spellchecker import SpellChecke



#loading of the csv
df = pd.read_csv(r"C:\Users\Swazy\agriculture_ie.csv")

print('test')                 




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
nlp = spacy.load('en_core_web_sm')

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

#df['clean_text'] = [' '.join(map(str, l)) for l in df['text']]



df = df.dropna()

#changing tokens into strings


print('t')

#processed file save
df.to_csv("C:/Users/Swazy/agriculture_ie_clean.csv")




