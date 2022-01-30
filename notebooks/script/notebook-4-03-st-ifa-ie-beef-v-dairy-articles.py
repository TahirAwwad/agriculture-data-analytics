#!/usr/bin/env python
# coding: utf-8

from bokeh.io import output_notebook
from IPython.core.interactiveshell import InteractiveShell
from nltk.corpus import stopwords
import cufflinks
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import warnings 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction import DictVectorizer


from scipy import stats ## Needed for stats test
import statsmodels.api as sm
from wordcloud import WordCloud, STOPWORDS

from matplotlib import pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')
InteractiveShell.ast_node_interactivity = 'all'
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)


#warnings.filterwarnings('ignore')


output_notebook()


cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

nlp = spacy.load('en_core_web_sm')


nltk.download('words')


words = set(nltk.corpus.words.words())


filename: str = "./../artifacts/ifa-ie-articles.csv"
df = pd.read_csv(filename)

df.columns 


df.info()


df.head(5)


def vader_scorer(df):
    '''Compute vaderSentiment scores for each article
    Args: Dataframe containing a 'text' column
    Returns: Dataframe of vader scores
    '''
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = df.loc[:,'Text'].map(analyzer.polarity_scores)

    dvec = DictVectorizer()
    vader_scores = dvec.fit_transform(vader_scores)
    
    vader_scores = pd.DataFrame(vader_scores.toarray(),columns=dvec.feature_names_)
    return vader_scores


vader_scores = vader_scorer(df)
df = pd.concat([df,vader_scores], axis=1)


df['compound'][df['Trend'] == 'cattle'].iplot(
    kind='hist',
    bins=50,
    xTitle='Sentiment Score',
    linecolor='black',
    yTitle='count',
    title='Token Polarity Distribution Of The Cattle Articles')


df['compound'][df['Trend'] == 'dairy'].iplot(
    kind='hist',
    bins=50,
    xTitle='Sentiment Score',
    linecolor='black',
    yTitle='count',
    title='Token Polarity Distribution Of The Dairy Articles')


df.info()


df.head(5)


fig, ax = plt.subplots(ncols=2, figsize=(12,4))
df[df.Trend=='cattle']['compound'].hist(ax=ax[0], bins=50)
df[df.Trend=='dairy']['compound'].hist(ax=ax[1], bins=50)
ax[0].set_xlabel('Cattle')
ax[1].set_xlabel('Dairy')
plt.show()


# Hypothesis:
# Null Hypothesis is that sentiment for the beef and the dairy was the same

fig, ax = plt.subplots(ncols=2, figsize=(10,4))
sns.violinplot(y='compound', x='Trend', data=df[df.Trend != ''], ax=ax[0])
sns.boxplot(y='compound', x='Trend', data=df[df.Trend != ''], ax=ax[1])
plt.tight_layout()
plt.show()


np.mean(df[df.Trend == 'cattle']['compound'])
np.mean(df[df.Trend == 'dairy']['compound'])


print('StDev of beef sentiment',np.std(df[df.Trend == 'cattle']['compound']))
print('StDev of dairy sentiment',np.std(df[df.Trend == 'dairy']['compound']))


# Running SciPy's t-test:




stats.ttest_ind(df[df.Trend == 'cattle']['compound'],
                df[df.Trend == 'dairy']['compound'], equal_var=True)


# Alternatively could have run Statsmodel API's ztest:

sm.stats.ztest(df[df.Trend == 'cattle']['compound'],
               df[df.Trend == 'dairy']['compound'])


# So p-value is 3.6%, which says that if null hypothesis is assumed to be true, then there is 3.62% chance of observing what we've just observed as the alternate hypothesis.If the null was true there is no chance that the alternative hypthesis is true.
# The 95% confidence interval can be constructed

p_bf = np.mean(df[df.Trend == 'cattle']['compound'])
p_dr = np.mean(df[df.Trend == 'dairy']['compound'])

num_bf = len(df[df.Trend == 'cattle'])
num_dr = len(df[df.Trend == 'dairy'])


# The s.e. for each population is sigma / sqrt(n)

se_bf = np.std(df[df.Trend == 'cattle']['compound']) / np.sqrt(num_bf)
se_dr = np.std(df[df.Trend == 'dairy']['compound']) / np.sqrt(num_dr)


# Alternatively, statsmodels has a standard error function of the mean of a distribution:

print('beef sentiment s.e.', stats.sem(df[df.Trend == 'cattle']['compound'], axis=None))
print('dairy sentiment s.e.', stats.sem(df[df.Trend == 'dairy']['compound'], axis=None))


se_diff = np.sqrt(se_bf**2 + se_dr**2) # Was se_br**2 + se_dr**2


diff = p_bf - p_dr
lcb = diff - (1.96 * se_diff)
ucb = diff + (1.96 * se_diff)
(lcb, ucb)


# Visualise sentiment
# Can we visualise what type of sentiment ifi had for the beef vs the dairy? Let's look at the words that were being used to better understand how the ifi described each entity:




beef_text = " ".join(art for art in df.clean_text [df.Trend=='cattle'][co]) # Changed to cattle
dairy_text = " ".join(art for art in df.clean_text[df.Trend=='dairy'])

stopwords = set(STOPWORDS)
stopwords.update(['http', 'https', 'www', 'amp', 'ly', 'bit','beef','steer','heifer',
                  'farmer','ifa','member','livestock','chairman','irish',
                  'cattle','say','brendan','golden','bull','young','angus','wood',
                  'joe', 'healy','milk','dairy','tom','phelan','week','year','month','cow'])

beef_wordcloud = WordCloud(stopwords=stopwords).generate(beef_text)
dairy_wordcloud = WordCloud(stopwords=stopwords).generate(dairy_text)

fig, ax = plt.subplots(nrows=2, figsize=(10,10))
ax[0].imshow(beef_wordcloud)
ax[0].set_title('beef')
ax[0].axis('off')
ax[1].imshow(dairy_wordcloud)
ax[1].set_title('dairy')
ax[1].axis('off')
plt.tight_layout()
plt.show


sns.distplot(df[df.Trend == 'cattle']['compound'], label='cattle')
sns.distplot(df[df.Trend == 'dairy']['compound'], label='dairy')
plt.legend()
plt.show()




