#!/usr/bin/env python
# coding: utf-8

# <style>
# *
# {
# 	text-align: justify;
# 	line-height: 1.5;
# 	font-family: "Arial", sans-serif;
# 	font-size: 12px;
# }
# 
# h2, h3, h4, h5, h6
# {
# 	font-family: "Arial", sans-serif;
# 	font-size: 12px;
# 	font-weight: bold;
# }
# h2
# {
# 	font-size: 14px;
# }
# h1
# {
# 	font-family: "Wingdings", sans-serif;
# 	font-size: 16px;
# }
# </style>

# ## Bovine Tuberculosis Herd Rate Prediction Model Builder

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-5-01-ml-irish-bovine-tuberculosis.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-5-01-ml-irish-bovine-tuberculosis.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-5-01-ml-irish-bovine-tuberculosis.ipynbynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective
# Create a machine learning model to predict rate of Herd Tuberculosis.

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from agriculture_data_analytics.project_manager import *
from agriculture_data_analytics.dataframe_labels import *
from keras_tuner.tuners import RandomSearch
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor
from data_analytics import github
import numpy as np
import os
import pandas
import pickle
import shutil


pandas.options.display.float_format = '{:.5f}'.format


READ_BINARY = "rb"
WRITE_BINARY = "wb"


model_directory: str = "county-bovine-tb-models"
directory: str = f'./../artifacts/{model_directory}/'


artifact_manager: ProjectArtifactManager = ProjectArtifactManager()
artifact_manager.is_remote = True
github.display_jupyter_notebook_data_sources(
    [artifact_manager.get_county_bovine_tuberculosis_eda_filepath()])
artifact_manager.is_remote = False


# ### Load dataframe

dataframe_filepath: str = artifact_manager.get_county_bovine_tuberculosis_eda_filepath()
dataframe: DataFrame = read_csv(dataframe_filepath)


print("Row, Column Count:", dataframe.shape)


dataframe


# ### Check the types for machine learning

dataframe.dtypes


# Veterinary Office is an object, specifically a string. We must encode it as a number for machine learning.

dummy_values_dataframe = dataframe[[
    "Veterinary Office"
]]

dataframe.drop('Veterinary Office',
                                          axis=1,
                                          inplace=True)

dummy_values_dataframe = pandas.get_dummies(dummy_values_dataframe,
                                            columns=["Veterinary Office"],
                                            prefix=["Veterinary Office "])


dataframe = dataframe.join(dummy_values_dataframe)


# ### Set Year as Index

dataframe.set_index(YEAR, drop=True, inplace=True)
dataframe.head()


dataframe.dtypes


dataframe.isnull().sum()


print("dataset dimensions", dataframe.shape)


# ### Prepare Model Data

# #### Select the feature set and the target

# Select the feature set and the target 

feature_values = dataframe.drop(columns=['Herd Incidence Rate']).values
target_values = dataframe['Herd Incidence Rate'].values.reshape(-1, 1)

print(f'Features dimension: {np.shape(feature_values)[1]} Columns, {np.shape(feature_values)[0]} Rows')
print(f'Target dimension:   {np.shape(target_values)[1]} Column,  {np.shape(target_values)[0]} Rows')


feature_values


target_values


target_values


# #### Define Training and Test Sets

# The data is liner data and should not be shuffled. Set test set size to 20%.

test_size: float = 0.8
X_train, X_test, Y_train, Y_test = train_test_split(feature_values,
                                                    target_values,
                                                    test_size=test_size,
                                                    shuffle=False)


# #### Scale & Transform

features_scaler = MinMaxScaler()

features_scaler.fit(X_train)

xtest_scale = features_scaler.transform(X_test)
xtrain_scale = features_scaler.transform(X_train)


target_scaler = MinMaxScaler()

target_scaler.fit(Y_train)

ytrain_scale = target_scaler.transform(Y_train)
ytest_scale = target_scaler.transform(Y_test)


# ##### Save Scalers

filename: str = 'features-scaler.pickle'
features_scaler_filepath: str = f'{directory}{filename}'

with open(features_scaler_filepath, WRITE_BINARY) as file:
    pickle.dump(features_scaler, file)


filename: str = 'target-scaler.pickle'
target_scaler_filepath: str = f'{directory}{filename}'

with open(target_scaler_filepath, WRITE_BINARY) as file:
    pickle.dump(target_scaler, file)


# ### Model Testing

# #### Score Board

model_scores_dataframe = DataFrame(
    columns=['Model', 'Mean Absolute Error Score(%)'])


# #### Random Forest Regressor

# ##### Train Model

# Hyper parameter tuning via Grid Search Cross Validation

xtrain_scale


random_forest_regressor = RandomForestRegressor()

random_forest_regressor_paramaters_grid = {
    'bootstrap': [True, False],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'max_depth': [1, 3, 5],
    'max_features': ["auto", "sqrt", "log2"],
    'n_estimators': [100, 500, 800],  # Number of trees
}

grid_search_cv = GridSearchCV(
    estimator=random_forest_regressor,
    param_grid=random_forest_regressor_paramaters_grid,
    n_jobs=-1,  # Use all processors on CPU
    cv=5)  #cross validation 5 fold of datasets

grid_search_cv.fit(xtrain_scale, ytrain_scale.reshape(-1))


# ##### Test and Score Model

print('Best training model', grid_search_cv.best_estimator_)
print('Best training model score, coefficient of determination R squared',
      grid_search_cv.best_score_)

y_predict = target_scaler.inverse_transform(
    grid_search_cv.predict(xtest_scale).reshape(-1, 1))
mae_score = mean_absolute_error(Y_test, y_predict)


# ##### Save Model Score

values = ['Random Forest Regressor', mae_score]
model_scores_dataframe.loc[len(model_scores_dataframe)] = values
model_scores_dataframe.head()


# #### XGBOOST Regressor

# ##### Train Model

# define XGBRegressor
xgb_regressor_milk = XGBRegressor(random_state=2021)

# define parameters space to loop over
params_xgb_milk = {
    'n_estimators': [20, 40, 80, 160, 340, 500],
    'max_depth': [3, 6, 9],
    'gamma': [0.01, 0.1],
    'learning_rate': [0.001, 0.01, 0.1, 1]
}

# Hyper parameter tuning via Grid Search Cross Validation
grid_xgb_milk = GridSearchCV(
    estimator=xgb_regressor_milk,
    param_grid=params_xgb_milk,
    #n_jobs=-1,
    scoring=['r2', 'neg_root_mean_squared_error'],
    refit='r2',
    n_jobs=-1,
    cv=5,
    verbose=4)

# fit grid to training scaled set
grid_xgb_milk.fit(xtrain_scale, ytrain_scale)

# print best training model & R squared score
print('Best training model ', grid_xgb_milk.best_estimator_)
print('Best model Parameters', grid_xgb_milk.best_params_)
print('Best training model score, coefficient of determination R squared',
      grid_xgb_milk.best_score_)


# ##### Test and Score Model

y_predict = target_scaler.inverse_transform(
    grid_xgb_milk.predict(xtest_scale).reshape(-1, 1))

mae_score = mean_absolute_error(Y_test, y_predict)


# ##### Save Model Score

model_scores_dataframe.loc[len(model_scores_dataframe)] = ['XGBOOST', mae_score]
model_scores_dataframe.head()


# #### ANN Artificial Neural Network

# ##### Train Model

#Training & Keras Parameter Tuning

temp_directory: str = './../temp/ANN-tuner/'


# Define ANN model with Hyper parameter variable
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


if os.path.isdir(temp_directory):
    try:
        shutil.rmtree(temp_directory)
    except OSError as exception:
        print(f"Error: {exception.filename} - {exception.strerror}.")

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
#Clean up
if os.path.isdir(temp_directory):
    try:
        shutil.rmtree(temp_directory)
    except OSError as exception:
        print(f"Error: {exception.filename} - {exception.strerror}.")


# ##### Test and Score Model

# Predict Milk Production and un-scale back to original values

y_predict = target_scaler.inverse_transform(
    bestANNModel.predict(xtest_scale).reshape(-1, 1))

print('predicted milk production values \n', y_predict)
print('actual milk production values \n', Y_test)

# Calculate Mean Absolute Error
mae_score = mean_absolute_error(Y_test, y_predict)
#print(MAE_xgb)


# ##### Save Model Score

model_scores_dataframe.loc[len(model_scores_dataframe)] = ['ANN', mae_score]
model_scores_dataframe.head()


# ### Save Artifacts

# Save trained model into binary pickle file to use the model later with new input data from web app

# #### Save Models

filename: str = 'ann-model.h5'
ann_filepath: str = f'{directory}{filename}'
bestANNModel.save(ann_filepath, save_format='h5')

