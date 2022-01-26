#!/usr/bin/env python
# coding: utf-8

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-3-03-ml-milk-production.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-3-03-ml-milk-production.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-3-03-ml-milk-production.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# # Statistical Analysis
# # Objectives
#        - Implement Statistical analysis on Milk Production value.
#        - Data: Value at Current Prices for Output, Input and Income in Agriculture
#        - Source: Central Statistics office CSO, Downloaded https://data.cso.ie/table/AEA01

# Hypothesis Testing: 
#     
#     - H0 Null Hypotheseis: There is no difference between sample set and popuation.
#     - H1 Alternative : There is a difference between sample set and population.
#     - reject H0 if we have enough statistical evidence
#     - enough statistical evidence p-value less than threshhold of 0.05
#     - p-value : probability the result is due to chance and not a general result
#     - the less p-value the better

# ## Import required library

import pandas as pd
#!pip install pingouin
#!pip install plotly
import pingouin as pg
import scipy.stats as stats
import math
import plotly.express as px


# ## Read Data

df = pd.read_csv("./../artifacts/TA_inputoutputvalue_1990_2021_CSO.csv")

## Extract milk production dataset
# drop redundunt columns
df = df.drop('Unnamed: 0',axis = 1)

# extract milk dataset
df = df[['Year',
#              'UNIT',
              'All Livestock Products - Milk',
              'Taxes on Products',
              'Subsidies on Products',
              'Compensation of Employees',
              'Contract Work',
              'Entrepreneurial Income',
              'Factor Income',
              'Fixed Capital Consumption - Farm Buildings',
              'Fixed Capital Consumption - Machinery, Equipment, etc',
              'Interest less FISIM',
              'Operating Surplus',
              'Livestock - Cattle',
              'Livestock - Sheep',
              'Land Rental',
              'Intermediate Consumption - Contract Work',
              'Intermediate Consumption - Crop Protection Products',
              'Intermediate Consumption - Energy and Lubricants',
              'Intermediate Consumption - Feeding Stuffs',
              'Intermediate Consumption - Fertilisers',
              'Intermediate Consumption - Financial Intermediation Services Indirect',
              'Intermediate Consumption - Forage Plants',
              'Intermediate Consumption - Maintenance and Repairs',
              'Intermediate Consumption - Seeds',
              'Intermediate Consumption - Services',
              'Intermediate Consumption - Veterinary Expenses',
              'Intermediate Consumption - Other Goods (Detergents, Small Tools, etc)',
              'Intermediate Consumption - Other Goods and Services'
              
             ]]
# Assign year as index
df.set_index('Year',drop=True,inplace=True)
# rename column
df = df.rename(columns={'All Livestock Products - Milk':'Milk Production'})


#print("Milk production dataset dimenssions \n", df_milk.shape,'\n')
#print( df_milk.info())

# chosse only Milk production column
df_milk = df['Milk Production']
fig = px.bar(df_milk,title='Irish Milk Production Value over Time - Thousands Euro')
fig.show()


# ## Descriptive Statistics

# Tendency and Spread
print('Summary Statistics\n\n', df_milk.describe())
print('\nSkewness level' ,round(df_milk.skew(),2))
print('\nKurtoses level',round(df_milk.kurt(),2))

#px.box(df_milk,title='Distribution')

px.violin(df_milk,box=True,points='all',title='Milk production Value Distribution 1990-2021')


fig = px.ecdf(df_milk,ecdfnorm='percent'
             ,title='Milk production Value Empirical Comulative Distribution 1990-2021')
fig.show()


# ## Normality Test

# Test Normality via Shapiro-Wilk test of Normality
print('Shapiro-Wilk test of Normality \n',
      pg.normality(df_milk),
      '\n')

# test with kruswales independece test
print('Kruswales independece test of our data and a normal distribution \n',
      stats.ks_1samp(df['Milk Production'],stats.norm.cdf),
      '\n')
# null hypothesis is that the two distributions are identical


# Test Normality via QQplot
print('Visual normality test \n')
ax = pg.qqplot(df_milk, dist='norm')


#     - Shapiro-Wilk test reported p-value of 0.00004
#     - p-value < aplha value of 0.05
#     - we have enoughe statistical evidence to reject Null hypothesis Milk production is normaly distributed
#     - QQplot confirms non-Normal distribution values with outliers
#     - NonParamatric tests are recomended to be perofrmed

# IF our dataset had a normal distribution we would use the 1 sample t-test
#stats.ttest_1samp(df_milk.iloc[25:], # sample data
#                 df_milk.iloc[20:25].mean()) # population mean)


# Mann-Whitney-U test for nonparametric distributions meaning it does not assume normality of the dataset 

# question: average production of years 2015 to 2021 is greater than average production of years 2009 to 2014
stats.mannwhitneyu(df_milk.iloc[25:],df_milk.iloc[19:25],alternative='greater')


# ## Correlation

# Correlation over the years or patern
px.line(df.iloc[:,0:4])

# correlaton non parametric (spearman)
pg.corr(df['Milk Production'], df['Subsidies on Products'],method='spearman').round(3)




