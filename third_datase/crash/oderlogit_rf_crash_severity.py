#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:31:44 2023

"""

import numpy as np
import pandas as pd
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor





import seaborn as sns
import matplotlib.pyplot as plt
from pdpbox import pdp


import statsmodels.api as sm
from statsmodels.api import MNLogit
from scipy.stats import norm






#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------
# 1- Read Data
Data = pd.read_csv('Data_all_years_normalised.csv')
print("------------Data information----------------")
print (Data.info())
# Features
#: Labels: MI =0 : SI =1 : Fatal =2
# select no-fatal records
Label = Data ['Severity']
index = np.where(Label!=2)
Data= Data.iloc[index]

features_name=['Year', 'Area_Speed', 'ACCLOC_X', 'ACCLOC_Y', 'Age', 'AM_Peak', 
               'PM_Peak','Stats_Area', 'Suburb', 'Month', 'Day', 'Position_Type', 
               'Horizontal_Align', 'Vertical_Align', 'Moisture_Cond', 
               'Weather_Cond', 'DayNight', 'Traffic_Ctrls', 'Unit_Type', 
               'Sex', 'Unit_Movement']




# 3- Fit the ordered logit
formula = 'Severity ~ Stats_Area+Suburb+Month+Day+Position_Type+ Horizontal_Align+Vertical_Align+Moisture_Cond+Weather_Cond+DayNight+Traffic_Ctrls+Unit_Type+Sex+Unit_Movement+Year+Area_Speed++ACCLOC_X+ACCLOC_Y+Age+AM_Peak+PM_Peak'


logit_model_ordered= MNLogit.from_formula(formula, data=Data)
result_ordered = logit_model_ordered.fit()

# Calculate the pseudo R-squared value 
nullmod_ordered= MNLogit.from_formula("Severity ~ 1", data=Data)
nullmod_result_ordered= nullmod_ordered.fit()
pseudo_r_squared= 1 - result_ordered.llf / nullmod_result_ordered.llf
print("Pseudo R-squared for pedestrian accidents:", pseudo_r_squared)

# Print the summary of the ordered logit model 
print(result_ordered.summary())

print("------------------------------------------------------------------------")
print("-------------------------------RF Results--------------------------------")

#-4-RF
# Create the feature matrix and target vector
X = Data[features_name]
y = Data['Severity']

# Fit the random forest model
rf_model = RandomForestRegressor(n_estimators=128, random_state=42)
rf_model.fit(X, y)

# Print the summary of the random forest model
print(rf_model)

# Get feature importances
importance = rf_model.feature_importances_

# Create a dataframe with feature importance values
rf_performance_eva = pd.DataFrame({'Feature': features_name, 'Importance': importance})
rf_performance_eva['MeanDecreaseAccuracy'] = rf_performance_eva['Importance']
rf_performance_eva.loc[rf_performance_eva['MeanDecreaseAccuracy'] < 0, 'MeanDecreaseAccuracy'] = 0
rf_performance_eva['relative_imp'] = rf_performance_eva['MeanDecreaseAccuracy'] / rf_performance_eva['MeanDecreaseAccuracy'].sum()
rf_performance_eva = rf_performance_eva.sort_values(by='relative_imp', ascending=False)
print(rf_performance_eva)
print("---------------------RElative importance---------------------")
for feature, importance_score in zip(features_name, importance):
    print(feature, ":", importance_score*100)


# 5-Plot partial dependence for 'Age'
plt.figure()
plt.plot(X['Age'], rf_model.predict(X), 'b.')
plt.xlabel('Age')
plt.ylabel('Partial Dependence')
plt.title('Partial Dependence Plot for Age')
plt.show()





