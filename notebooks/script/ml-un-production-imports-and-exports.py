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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


total_df = pd.read_csv ('combined.csv')
total_df


ratio_df = pd.read_csv ('combined_ratio.csv')
ratio_df





# # Machine Learning

# ## Data pre processing

#Create the machine learning dataframe and transform Area and Items to dummie variables
ml_df = total_df[["Area","Item"]]
ml_df = pd.get_dummies(ml_df,columns=["Area","Item"])
ml_df.reset_index(inplace=True,drop=True)
ml_df.tail(2)


#Scale this columns and join into ml_df
columns_to_scale = ["Import_Qt_ton","Export_Qt_ton","Agri_1000ha","Production_ton"]
scaler = MinMaxScaler()
ml_df = ml_df.join(pd.DataFrame(scaler.fit_transform(total_df[columns_to_scale]),columns=columns_to_scale))
ml_df.head()


X_train, X_test,y_train,y_test = train_test_split(ml_df.iloc[:,:-1],ml_df.iloc[:,-1],test_size=0.2,random_state=42)


print("X Train size:",X_train.shape[0],
      "\ny Train size:",y_train.shape[0],
      "\nX Test size:",X_test.shape[0],
      "\ny Test size:",y_test.shape[0])





# ## Random Forest Regressor

# define list of Parameters
params = {'n_estimators':[50,100,300,500,800,1000],
                  "min_samples_split":[2,5,10,20,40],
                  "max_depth":[None,5,10,20,40,50]
                   }

# Hyper parameter tuning via Grid Search Cross Validation 
grid_rf = GridSearchCV(estimator= RandomForestRegressor(random_state=42),
                          param_grid= params,
                          n_jobs=-1,
                          cv=5,
                          verbose=4,
                          refit= True
                     )

# Fit
grid_rf.fit(X_train,y_train)

# print best training model & score
print('Best training model:',grid_rf.best_estimator_)
print('Best model Parameters',grid_rf.best_params_)
print('Best training model score:', grid_rf.best_score_)


# Predict Production and unscale back to original values
y_pred = grid_rf.predict(X_test)

# Calculate Mean Absolute Error
MAE_rf = mean_absolute_error(y_test,y_pred)
print(MAE_rf)





# ## XGBoost Regressor

# define parameters space to loop over
params = {'n_estimators':[20,40,80,160,340,500],
             'max_depth':[3,6,9],
             'gamma':[0.01,0.1],
             'learning_rate':[0.001,0.01,0.1,1]
             }

# Hyper parameter tuning via Grid Search Cross Validation 
grid_xgb = GridSearchCV(estimator=XGBRegressor(random_state=42),
                     param_grid=params,
                     refit= True,
                     n_jobs=-1,
                     cv=5,
                     verbose=4
                     )

# fit grid to training scaled set
grid_xgb.fit(X_train,y_train)


# print best training model & R squared score
print('Best training model ',grid_xgb.best_estimator_)
print('Best model Parameters',grid_xgb.best_params_)
print('Best training model score, coefficient of determination R squared', grid_xgb.best_score_)


# Predict Production and unscale back to original values
y_pred = grid_xgb.predict(X_test)

# Calculate Mean Squared Error
MAE_rf = mean_absolute_error(y_test,y_pred)
print(MAE_rf)





# ## Neural Network

X_train.shape


#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model = Sequential()
model.add(Dense(32, input_dim=16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=10)


# Predict Production and unscale back to original values
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
MAE_rf = mean_absolute_error(y_test,y_pred)
print(MAE_rf)







