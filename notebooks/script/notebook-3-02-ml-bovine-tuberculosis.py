#!/usr/bin/env python
# coding: utf-8

# ## Bovine Tuberculosis

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-3-02-ml-bovine-tuberculosis.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-3-02-ml-bovine-tuberculosis.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-3-02-ml-bovine-tuberculosis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to build a machine learning (ML) model of the [bovine-tuberculosis-eda-output.csv](./../artifacts/bovine-tuberculosis-eda-output.csv).

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt --quiet --user
# Remote option
#!pip install -r https://github.com/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/script/requirements.txt --quiet --user


from agriculture_data_analytics.project_manager import *
from agriculture_data_analytics.dataframe_labels import *
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import data_analytics.github as github
import numpy
import os
import pandas


artifact_manager: ProjectArtifactManager = ProjectArtifactManager()
artifact_manager.is_remote = True
github.display_jupyter_notebook_data_sources(
    [artifact_manager.get_county_bovine_tuberculosis_eda_filepath()])
artifact_manager.is_remote = False


# ### Load dataframe

filepath: str = artifact_manager.get_county_bovine_tuberculosis_eda_filepath()
county_bovine_tuberculosis_dataframe: DataFrame = pandas.read_csv(filepath)


print("Row, Column Count:", county_bovine_tuberculosis_dataframe.shape[0])


# ### Check the types for machine learning

county_bovine_tuberculosis_dataframe.dtypes


# Veterinary Office is an object, specifically a string. We must encode it as a number for machine learning.

dummy_values_dataframe = county_bovine_tuberculosis_dataframe[[
    "Veterinary Office"
]]

county_bovine_tuberculosis_dataframe.drop('Veterinary Office',
                                          axis=1,
                                          inplace=True)

dummy_values_dataframe = pandas.get_dummies(dummy_values_dataframe,
                                            columns=["Veterinary Office"],
                                            prefix=["Veterinary Office "])


county_bovine_tuberculosis_dataframe = county_bovine_tuberculosis_dataframe.join(dummy_values_dataframe)


# ### Set Year as Index

county_bovine_tuberculosis_dataframe.set_index(YEAR, drop=True, inplace=True)
county_bovine_tuberculosis_dataframe.head()


county_bovine_tuberculosis_dataframe.dtypes


county_bovine_tuberculosis_dataframe.isnull().sum()


# ### Define 20% Training set 80% Test set

# define target & feature variables

X = county_bovine_tuberculosis_dataframe.iloc[:, 2:].values
Y = county_bovine_tuberculosis_dataframe.iloc[:, 1].values.reshape(-1)
print(numpy.shape(X))
print(numpy.shape(Y))

# split train test split 20
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=2021)


# ### Model 1 RandomForest Regressor

random_forest_regressor = RandomForestRegressor(random_state=2021)


random_forest_params = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt']
}


grid_search_cv = GridSearchCV(estimator=random_forest_regressor,
                              param_grid=random_forest_params)


# np.isnan(X_train).sum()
# np.nan_to_num(X_train)
# np.nan_to_num(Y_train)
grid_search_cv.fit(X_train, Y_train)  #do not run becuase of null values


# print best model
print(grid_search_cv.best_estimator_)
print('Best model score', grid_search_cv.best_score_)


# ### Model 2 XGBOOST Regressor

xgb_regressor = XGBRegressor(random_state=2021)


# make a search space of parameters to loop over

xgb_regressor_params = {
    'n_estimators': [20, 40, 80, 160, 340, 500],
    'max_depth': [3, 6, 9],
    'gamma': [0.01, 0.1],
    'learning_rate': [0.001, 0.01, 0.1, 1]
}


grid_search_cv = GridSearchCV(
    estimator=xgb_regressor,
    param_grid=xgb_regressor_params,
    #n_jobs=-1,
    scoring=['r2', 'neg_root_mean_squared_error'],
    refit='r2',
    cv=5,
    verbose=4)


grid_search_cv.fit(X_train, Y_train)


# print best model
print(grid_search_cv.best_estimator_)


# print best parameters
print('Best model Parameters', grid_search_cv.best_params_)
# best score
print('Best model R2 score', grid_search_cv.best_score_)

# write the Grid Search results to csv to choose best model with least resource consumption
bovine_tuberculosis_xgb_grid_search_dataframe = DataFrame(
    grid_search_cv.cv_results_)
bovine_tuberculosis_xgb_grid_search_dataframe = bovine_tuberculosis_xgb_grid_search_dataframe.sort_values(
    'rank_test_r2')


bovine_tuberculosis_xgb_grid_search_dataframe.to_csv(
    './../artifacts/grid-search-xgb-county-bovine-tuberculosis-results.csv')

