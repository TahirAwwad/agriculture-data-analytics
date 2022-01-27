#!/usr/bin/env python
# coding: utf-8

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-3-03-ml-milk-production.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-3-03-ml-milk-production.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-3-03-ml-milk-production.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# # Objective
# Create the best machine learning model to predict Milk production value in Ireland according to the historical data from Central Statistics office CSO. AEA01 Value at Current Prices for Output, Input and Income in Agriculture Downloaded https://data.cso.ie/table/AEA01 

# # Contents
#     - Read data from Assets folder
#     - Split to training / testing sets
#     - Scale each set seperatly
#     - Run Models
#         - Define Hyper parameter tuning Cross Validation Grid or Random Search
#         - Random Forest Regressor
#         - XGBOOST Regressor
#         - ANN
#     - Save best model into Pickle file
#     - Next step: Deploy selected model on a Streamlit webapp

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from keras_tuner.tuners import RandomSearch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf


# ### Load dataframe

df = pd.read_csv("./../artifacts/irish-milk-production-eda-output.csv")
print("data dimensions \n",df.shape)
print()
#print("data column info \n",df.info)
print()
print('Data sample\n')

df.sample(5)


# ## Production of Milk

## Extract milk production dataset
# drop redundunt columns

# extract milk dataset
df_milk = df[['Year',
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
df_milk.set_index('Year',drop=True,inplace=True)

print("Milk production dataset dimenssions \n", df_milk.shape)
print("Milk production dataset Sample \n")
df_milk.head()


# ### Define 20% Training set 80% Test set

# define target & feature variables

X = df_milk.iloc[:,2:].values
Y = df_milk.iloc[:,1].values.reshape(-1,1)
print('features dimension ',np.shape(X))
print('target dimension ',np.shape(Y))

# impute mean value for NA's
#from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp_mean.fit_transform(X)
Y = imp_mean.fit_transform(Y)


# split train test split 20
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2021)
print()
print('x_train dimension ', X_train.shape)
print('y_train dimension ', Y_train.shape)
print()
print('x_test dimension ', X_test.shape)
print('y_test dimension ', Y_test.shape)


# ### Scale & Transform

# Scale raining set and test set seperatly
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# calculate mean and std of training set 
scaler_x.fit(X_train)
scaler_y.fit(Y_train)
scaler_x.fit(X_test)
scaler_y.fit(Y_test)

# apply scaler to data set
xtrain_scale = scaler_x.transform(X_train)
ytrain_scale = scaler_y.transform(Y_train)

xtest_scale = scaler_x.transform(X_test)
ytest_scale = scaler_y.transform(Y_test)

# fit and transform in one line
# scaler_x.fit_transform(X_train)

# remeber to inverse the scaling on model output
# scaler_x.inverse_transform(xtest_scale)

# create a score dataframe to store model scores
df_score = pd.DataFrame()
print(df_score)


# ### Model 1 RandomForest Regressor

# #### Train RandomForest

# define Random Forest Regressor
rf_regressor_milk = RandomForestRegressor(random_state=2021)

# define list of Parameters
params_rf_milk = {'n_estimators':[100,500,800],
                  'criterion':['squared_error', 'absolute_error', 'poisson'],
                  'max_features':["auto", "sqrt", "log2"],
                  "bootstrap": [True, False]
                   }

# Hyper parameter tuning via Grid Search Cross Validation 
grid_rf_milk = GridSearchCV(estimator= rf_regressor_milk,
                          param_grid= params_rf_milk,
                          n_jobs=-1,
                          cv=5
                     )

# Fit the grid to scaled data
grid_rf_milk.fit(xtrain_scale,ytrain_scale.reshape(-1))

# print best training model & R squared score
print('Best training model ',grid_rf_milk.best_estimator_)
print('Best training model score, coefficient of determination R squared', grid_rf_milk.best_score_)


# #### Predict RandomForest

# Predict Milk Production and unscale back to original values
y_predict = scaler_y.inverse_transform(grid_rf_milk.predict(xtest_scale).reshape(-1, 1))

print('predicted milk production values \n',y_predict)
print('actual milk production values \n',Y_test)

# Calculate Mean Absolute Error
MAE_rf = mean_absolute_error(Y_test,y_predict)
#print(MAE_rf)

# add model score to Score Dataframe
df_score = pd.DataFrame(data={'Model':'RandomForest',
                           'Score MAE':MAE_rf},index=['Model 1'])

print(df_score)


# ### Model 2 XGBOOST Regressor

# #### Train XGBOOST

# define XGBRegressor
xgb_regressor_milk = XGBRegressor(random_state=2021)

# define parameters space to loop over
params_xgb_milk = {'n_estimators':[20,40,80,160,340,500],
             'max_depth':[3,6,9],
             'gamma':[0.01,0.1],
             'learning_rate':[0.001,0.01,0.1,1]
             }

# Hyper parameter tuning via Grid Search Cross Validation 
grid_xgb_milk = GridSearchCV(estimator=xgb_regressor_milk,
                     param_grid=params_xgb_milk,
                     #n_jobs=-1,
                     scoring=['r2','neg_root_mean_squared_error'],
                     refit= 'r2',
                     n_jobs=-1,
                     cv=5,
                     verbose=4
                     )

# fit grid to training scaled set
grid_xgb_milk.fit(xtrain_scale,ytrain_scale);


# print best training model & R squared score
print('Best training model ',grid_xgb_milk.best_estimator_)
print('Best model Parameters',grid_xgb_milk.best_params_)
print('Best training model score, coefficient of determination R squared', grid_xgb_milk.best_score_)


# #### Predict XGBOOST

# Predict Milk Production and unscale back to original values
y_predict = scaler_y.inverse_transform(grid_xgb_milk.predict(xtest_scale).reshape(-1, 1))

print('predicted milk production values \n',y_predict)
print('actual milk production values \n',Y_test)

# Calculate Mean Absolute Error
MAE_xgb = mean_absolute_error(Y_test,y_predict)
#print(MAE_xgb)

# add model score to Score Dataframe
df_score = pd.DataFrame(data = {'Model':['RandomForest','XGBOOST'],
                                'Score MAE': [MAE_rf,MAE_xgb]},
                        index=['Model 1','Model 2'])

print(df_score)


# write the Grid Search results to csv to choose best model with least resource consumption

#GS_xgb_df_milk = pd.DataFrame(GS_xgb_milk.cv_results_)
#GS_xgb_df_milk = GS_xgb_df_milk.sort_values('rank_test_r2')
#GS_xgb_df_milk.to_csv('./../artifacts/grid-search-xgb-milk-results.csv')


# ## ANN Artificial Neural Network

# #### Training & Keras Parameter Tuning

temp_directory: str = './../temp/ANN-tuner/'


# Define ANN model with Hyper paramter variable
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 23)):
        model.add(
            layers.Dense(units=hp.Int('units_' + str(i),
                                      min_value=23,
                                      max_value=600,
                                      step=32),
                         activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])
        return model

#if os.path.isdir(temp_directory):
#    os.remove(temp_directory)


# create a directory to store each iteration of modelling
tuner = RandomSearch(build_model,
                     objective='val_mean_absolute_error',
                     max_trials=5,
                     executions_per_trial=3,
                     directory=temp_directory,
                     project_name='Milk production')

# Defined parameter space to search in
tuner.search_space_summary()

# train trial models and compare with validation set
tuner.search(xtrain_scale,
             ytrain_scale,
             epochs=50,
             validation_data=(xtest_scale, ytest_scale))

# print best 10 models according to val_mean_absolute_error
print('\n')
tuner.results_summary()

# get best model from training trials
bestANNModel = tuner.get_best_models(num_models=1)[0]

# fit best model to training scaled data and scaled test data
bestANNModel.fit(xtrain_scale,
                 ytrain_scale,
                 epochs=50,
                 validation_data=(xtest_scale, ytest_scale))


# Predict Milk Production and unscale back to original values
y_predict = scaler_y.inverse_transform(bestANNModel.predict(xtest_scale).reshape(-1, 1))

print('predicted milk production values \n',y_predict)
print('actual milk production values \n',Y_test)

# Calculate Mean Absolute Error
MAE_ANN = mean_absolute_error(Y_test,y_predict)
#print(MAE_xgb)

# add model score to Score Dataframe
df_score = pd.DataFrame(data = {'Model':['RandomForest','XGBOOST','ANN'],
                                'Score MAE': [MAE_rf,MAE_xgb,MAE_ANN]},
                        index=['Model 1','Model 2','Model 3'])

print(df_score)


# # Pickle file
#     Save trained model into binary pickle file to use the model later with new input data from web app

model_name = "milk-production"
directory = f'./../artifacts/{model_name}/'

# Dump/write Scaler into binary pickle
pickle.dump(scaler_x, open(f'{directory}pkl_scaler_x', 'wb'))

# Read pickle file into variable to use scaler
scaler_x_pkl_ann = pickle.load(open(f'{directory}pkl_scaler_x', 'rb'))

# Dump/write Scaler into binary pickle
pickle.dump(scaler_y, open(f'{directory}pkl_scaler_y', 'wb'))

# Read pickle file into variable to use scaler
scaler_y_pkl_ann = pickle.load(open(f'{directory}pkl_scaler_y', 'rb'))


# Dump/write model into binary pickle file in the current notebook directory
pickle.dump(bestANNModel, open(f'{directory}pkl_ann_milk', 'wb'))
# Read pickle file into variable to use model
model_pkl_ann = pickle.load(open(f'{directory}pkl_ann_milk', 'rb'))


## Example using pickle file with saved ANN model

# take input from source as array
data_input_from_webapp = np.array([357.3, 362.5, 172., 1440.2, 2136.5])

# scale input with same scaler as used in model
scale_data_from_webapp = scaler_x.transform(
    data_input_from_webapp.reshape(1, -1))

# predict scaled value
scaled_prediction = bestANNModel.predict(scale_data_from_webapp)

# descale prediction back to normal value
prediction = scaler_y.inverse_transform(scaled_prediction)
print('\n Expected Milk Production is ', prediction[0][0])

