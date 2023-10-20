#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:31:44 2023

"""

import numpy as np
import pandas as pd
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



import seaborn as sns
import matplotlib.pyplot as plt
from pdpbox import pdp


import statsmodels.api as sm
import statsmodels.formula.api as smf






#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------
# 1- Read Data
Data = pd.read_csv('All_years_Casul_Data_unNormalised.csv')
Data = Data.drop(columns= ['exact_Date'])
print("------------Data information----------------")
print (Data.info())
# Features
#: Labels: MI =0 : SI =1 : Fatal =2
Label = Data ['Severity']

features_name=['Year', 'Area_Speed', 'ACCLOC_X', 'ACCLOC_Y', 'Age', 'AM_Peak', 
               'PM_Peak','Stats_Area', 'Suburb', 'Month', 'Day', 'Position_Type', 
               'Horizontal_Align', 'Vertical_Align', 'Moisture_Cond', 
               'Weather_Cond', 'DayNight', 'Traffic_Ctrls', 'Unit_Type', 
               'Sex', 'Unit_Movement']

# 2-select Fatal and non Fatal
Y =  np.zeros((len(Label),1))
ind = np.where (Label == 2)
Y [ind] =1
Data['Severity'] =Y


# 3- Binomial Logit
formula = 'Severity ~ Stats_Area+Suburb+Month+Day+Position_Type+ Horizontal_Align+Vertical_Align+Moisture_Cond+Weather_Cond+DayNight+Traffic_Ctrls+Unit_Type+Sex+Unit_Movement+Year+Area_Speed++ACCLOC_X+ACCLOC_Y+Age+AM_Peak+PM_Peak'
logit_model_fatality= smf.glm(formula, family=sm.families.Binomial(),
                               data=Data)

print('hi')
# Fit the logistic regression model
logit_result = logit_model_fatality.fit()
# Print summary of the logistic regression model
print(logit_result.summary())

# Print McFadden's pseudo R-squared
nullmod_fatality = smf.glm('Severity ~ 1', family=sm.families.Binomial(), 
                               data=Data)
nullmod_result = nullmod_fatality.fit()
pseudo_r2 = 1 - (logit_result.llf / nullmod_result.llf)
print("McFadden's pseudo R-squared:", pseudo_r2)


# 4- Random Forest
# Fit the random forest model

for i in range(300):
   rf_model = RandomForestClassifier(n_estimators=128)
   rf_model.fit(Data[features_name], Y)
   if i ==0:
       importance = rf_model.feature_importances_
   else:
       importance = importance+ rf_model.feature_importances_
   

# Print summary of the random forest model
importance = importance/300

#Create a dataframe with feature importance values
rf_performance_eva = pd.DataFrame({'Feature': features_name, 'Importance': importance})
rf_performance_eva['MeanDecreaseAccuracy'] = rf_performance_eva['Importance']
rf_performance_eva.loc[rf_performance_eva['MeanDecreaseAccuracy'] < 0, 'MeanDecreaseAccuracy'] = 0
rf_performance_eva['relative_imp'] = rf_performance_eva['MeanDecreaseAccuracy'] / rf_performance_eva['MeanDecreaseAccuracy'].sum()
rf_performance_eva = rf_performance_eva.sort_values(by='relative_imp', ascending=False)
print(rf_performance_eva)

print("---------------------RElative importance---------------------")
for feature, importance_score in zip(features_name, importance):
    print(feature, ":", importance_score*100)


print("-------------------Feature Importance-------------------------")
for feature, importance_score in zip(features_name, importance):
    print(feature, ":", importance_score*100)


