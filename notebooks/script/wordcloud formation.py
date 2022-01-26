#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd


# Loading the text data
dataframe = pd.read_csv("./../artifacts/agriculture-ie-clean.csv")
text = str(dataframe['clean_text'])
print(text)
dataframe.info()
# Creating empty WordCloud object and generating actual WordCloud.
wordcloud = WordCloud().generate(text)
# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

