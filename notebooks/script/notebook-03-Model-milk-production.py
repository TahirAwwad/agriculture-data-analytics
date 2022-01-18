#!/usr/bin/env python
# coding: utf-8

# # Read full dataset of Agriculture Production since 1991 to 2020

import pandas as pd
import numpy as np


df = pd.read_csv("../assets/TA_inputoutputvalue_1990_2021_CSO.csv")
print("data dimensions \n",df.shape)
print()
print("data column info \n",df.info)


# # Production of Milk

## Extract milk production dataset
# drop redundunt columns
df = df.drop('Unnamed: 0',axis=1)

# extract milk dataset
df_milk = df[['Year',
              'UNIT',
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


df_milk = df[['Year',
              'UNIT',
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


# ## Define 20% Training set 80% Test set

# define target & feature variables

X = df_milk.iloc[:,2:].values
Y = df_milk.iloc[:,1].values.reshape(-1,1)
print(np.shape(X))
print(np.shape(Y))

# split train test split 20
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2021)


# ## Model 1 RandomForest Regressor

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


rf_model_milk = RandomForestRegressor(random_state=2021)


params_rf_milk = {'n_estimators':[100,200,500]
            ,'max_features':['auto','sqrt']
            }


GS_rf_milk = GridSearchCV(estimator= rf_model_milk,
                     param_grid= params_rf_milk
                     )


# np.isnan(X_train).sum()
# np.nan_to_num(X_train)
# np.nan_to_num(Y_train)
GS_rf_milk.fit(X_train,Y_train)


# print best model
print(GS_rf_milk.best_estimator_)
print('Best model score',GS_rf_milk.best_score_)


# ## Model 2 XGBOOST Regressor

# xgboost
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


GS_xgb_milk.fit(X_train,Y_train)


# print best model
print(GS_xgb_milk.best_estimator_)


# print best parameters
print('Best model Parameters',GS_xgb_milk.best_params_)
# best score
print('Best model R2 score',GS_xgb_milk.best_score_)

# write the Grid Search results to csv to choose best model with least resource consumption
GS_xgb_df_milk = pd.DataFrame(GS_xgb_milk.cv_results_)
GS_xgb_df_milk = GS_xgb_df_milk.sort_values('rank_test_r2')
GS_xgb_df_milk.to_csv('GS_xgb_milk_results.csv')


predict(X_test)




