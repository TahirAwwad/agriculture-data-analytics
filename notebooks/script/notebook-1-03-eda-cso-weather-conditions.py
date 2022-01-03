#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(rc = {'figure.figsize':(15,8)})
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random as rd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings


area_farmed = pd.read_csv('./../assets/area_farmed_june.csv')
land_cover = pd.read_csv('clc_land_.cover.csv')
fert = pd.read_csv('fertilizers.csv')
rain = pd.read_csv('rainfall.csv')
sun = pd.read_csv('sunshine.csv')
temp = pd.read_csv('temperature.csv')


area_farmed
#2023-2021


area_farmer = area_farmed.rename({'Type of Land Use': 'Land_Type'})














land_cover
#2012-2018, not yearly but reffering to change withing the gap.

















fert
#1980 - 2021, monthly.


fert = fert.rename(columns={"Type of Fertiliser": "Fertilizer_Type"})














rain


rain = rain.rename(columns={"Meteorological Weather Station": "Met_Weather_Station"})














sun


sun = sun.rename(columns={"Meteorological Weather Station": "Met_Weather_Station"})














temp


temp = temp.rename(columns={"Meteorological Weather Station": "Met_Weather_Station"})


temp.Statistic.unique()


temp.Met_Weather_Station.unique()













