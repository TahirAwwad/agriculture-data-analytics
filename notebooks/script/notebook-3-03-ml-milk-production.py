#!/usr/bin/env python
# coding: utf-8

# ## Read full dataset of Agriculture Production since 1991 to 2020

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-3-03-ml-milk-production.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-3-03-ml-milk-production.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-3-03-ml-milk-production.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
get_ipython().system('pip install -r script/requirements.txt --quiet --user')
# Remote option
get_ipython().system('pip install -r https://github.com/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/script/requirements.txt --quiet --user')


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import data_analytics.exploratory_data_analysis as eda
#import data_analytics.exploratory_data_analysis_reports as eda_reports


# ### Load dataframe

df = pd.read_csv("./../artifacts/TA_inputoutputvalue_1990_2021_CSO.csv")
print("data dimensions \n",df.shape)
print()
print("data column info \n",df.info)


df.head()
#eda_reports.print_dataframe_analysis_report(df)


# ## Production of Milk

## Extract milk production dataset
# drop redundunt columns
df = df.drop('Unnamed: 0',axis = 1)

# extract milk dataset
df_milk = df[['Year',
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
              #'Intermediate Consumption - Services',
              'Intermediate Consumption - Veterinary Expenses',
              'Intermediate Consumption - Other Goods (Detergents, Small Tools, etc)',
              #'Intermediate Consumption - Other Goods and Services'
              
             ]]
# Assign year as index
df_milk.set_index('Year',drop=True,inplace=True)

print("Milk production dataset dimenssions \n", df_milk.shape)


#eda_reports.print_dataframe_analysis_report(df_milk)


# ### Define 20% Training set 80% Test set

# define target & feature variables

X = df_milk.iloc[:,2:].values
Y = df_milk.iloc[:,1].values.reshape(-1,1)
print(np.shape(X))
print(np.shape(Y))

# impute mean value for NA
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp_mean.fit_transform(X)
Y = imp_mean.fit_transform(Y)


# split train test split 20
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2021)


scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)

print(scaler_y.fit(Y_train))
ytrain_scale=scaler_y.transform(Y_train)

print(scaler_x.fit(X_test))
xtest_scale=scaler_x.transform(X_test)

print(scaler_x.fit(Y_test))
ytest_scale=scaler_y.transform(Y_test)


# fill NAN values with the average  mean scaled
np.isnan(np.sum(xtrain_scale))
xtrain_scale[np.isnan(xtrain_scale)==True]= np.nanmean(xtrain_scale)
np.isnan(np.sum(ytrain_scale))
ytrain_scale[np.isnan(ytrain_scale)==True]= np.nanmean(ytrain_scale)
np.isnan(np.sum(xtest_scale))
xtest_scale[np.isnan(xtest_scale)==True]= np.nanmean(xtest_scale)
np.isnan(np.sum(ytest_scale))
ytest_scale[np.isnan(ytest_scale)==True]= np.nanmean(ytest_scale)


# ### Model 1 RandomForest Regressor

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


rf_model_milk = RandomForestRegressor(random_state=2021)


params_rf_milk = {'n_estimators':[100,200,500],
                  'criterion':['squared_error', 'absolute_error', 'poisson'],
                  'max_features':["auto"]
                  
            }


GS_rf_milk = GridSearchCV(estimator= rf_model_milk,
                     param_grid= params_rf_milk
                     )


GS_rf_milk.fit(xtrain_scale,ytrain_scale.reshape(-1))


# print best model
print(GS_rf_milk.best_estimator_)
print('Best model score', GS_rf_milk.best_score_)


# ### Model 2 XGBOOST Regressor

# xgboost
get_ipython().system('pip install xgboost')


from xgboost import XGBRegressor
xgb_model_milk = XGBRegressor(random_state=2021)


# make a search space of parameters to loop over

params_xgb_milk = {'n_estimators':[20,40,80,160,340,500],
             'max_depth':[3,6,9],
             'gamma':[0.01,0.1],
             'learning_rate':[0.001,0.01,0.1,1]
             }


GS_xgb_milk = GridSearchCV(estimator=xgb_model_milk,
                     param_grid=params_xgb_milk,
                     #n_jobs=-1,
                     scoring=['r2','neg_root_mean_squared_error'],
                     refit= 'r2',
                     cv=5,
                     verbose=4
                     )


GS_xgb_milk.fit(xtrain_scale,ytrain_scale);


# print best model
print(GS_xgb_milk.best_estimator_)


# print best parameters
print('Best model Parameters',GS_xgb_milk.best_params_)
# best score
print('Best model R2 score',GS_xgb_milk.best_score_)

# write the Grid Search results to csv to choose best model with least resource consumption
GS_xgb_df_milk = pd.DataFrame(GS_xgb_milk.cv_results_)
GS_xgb_df_milk = GS_xgb_df_milk.sort_values('rank_test_r2')


GS_xgb_df_milk.to_csv('./../artifacts/grid-search-xgb-milk-results.csv')


predict(X_test)


# ## ANN

#!pip install --upgrade tensorflow
import math
import matplotlib.pyplot as plt
#import numpy as np
from numpy.random import seed
seed(2021)
#import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler





model = Sequential()
# input layers  = Number of features in the training set + 1
model.add(Dense(24, input_dim=24, kernel_initializer='normal', activation='relu'))
# hidden layers = Training Data Samples/Factor * (Input Neurons + Output Neurons)
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=30, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(xtest_scale)


print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


predictions = scaler_y.inverse_transform(predictions)
predictions


# ## ANN hyper parameter tuning

#!pip install tensorflow
#!pip install -q -U keras-tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch


def build_model(hp):
    model= keras.Sequential()
    for i in range(hp.Int('num_layers',2,23)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                           min_value=23,
                                           max_value=600,
                                           step=32),
                              activation='relu'))
        model.add(layers.Dense(1,activation='linear'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
        return model


# create a directory to store each iteration of modelling
tuner = RandomSearch(
        build_model,
        objective='val_mean_absolute_error',
        max_trials=5,
        executions_per_trial=3,
        directory='CA2',
        project_name='Milk production forecast')


# parameter space to search in
tuner.search_space_summary()


# train the model
tuner.search(xtrain_scale,ytrain_scale,epochs=20,validation_data=(xtest_scale,ytest_scale))


# print best 10 models according to previously selected metric
tuner.results_summary()




