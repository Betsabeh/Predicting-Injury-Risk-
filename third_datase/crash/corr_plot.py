# -*- coding: utf-8 -*-
"""
@author: Betsabeh
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer


import seaborn as sns
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# **-fill the unknown with most frequent value in each column
def fill_missing_most_frequent(NewData, col_name, missing_val, strategy_method):  
     mostfreq_imputer = SimpleImputer(missing_values=missing_val,
                                      strategy=strategy_method)
    
     #NewData = Data.copy()
     num = np.shape(NewData[col_name])
     X = np.reshape(NewData[col_name].values, (num[0],1))
     temp = mostfreq_imputer.fit_transform(X)
     for i in range(len(temp)):
         NewData[col_name].iloc[i] = temp[i]
     #Data[col_name] =temp.tolist ()#temp.tolist()
     
     #print("here:,",Data[col_name])
     return NewData
#--------------------------------------------------------------------
#--------------------------------------------------------------------
def fill_all_missing(Data):
    # fill in missing values by the most common
    #NewData = Data.copy()  
    #print("hi")
    Data = fill_missing_most_frequent(Data, 'ACCLOC_X',
                                         0, 'mean')
    #print("hi2")
    Data = fill_missing_most_frequent(Data, 'ACCLOC_Y',
                                         0 ,'mean')

    return Data
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 1- Read Data
Data = pd.read_csv('All_years_Casul_Data.csv')
#Data = Data.drop(columns= ['REPORT_ID.1'])
print("------------Data information----------------")
print (Data.info())
num_samples = np.shape(Data['REPORT_ID'])[0]
print("number of samples =", num_samples)
Label = Data['CSEF Severity']
Data = Data.drop(columns= ['REPORT_ID', 'Casualty Type', 'CSEF Severity'])
# convert labels
le = LabelEncoder()
Numeric_labels = le.fit_transform(Label)
Labels_class = np.unique(Label)
count =[]
for class_name in Labels_class:
    ind = np.where (Label == class_name )
    count.append(np.shape(ind)[1])
    
fig1 = plt.Figure(figsize=(12,6))
plt.plot(count)
print("labels:", Labels_class)
print(count)

NewData = fill_all_missing(Data)
print("---------------------------------------------------------------------")
# 2-Select categorical and numeric column
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns_name = numerical_columns_selector(Data)
categorical_columns_name = categorical_columns_selector(Data)

# 3- ColumnTransformer 
num_cat_feature = len(categorical_columns_name)
categorical_transformed_features = np.zeros(shape =(num_samples,num_cat_feature))
le = LabelEncoder()
i=0
for name in categorical_columns_name:
    D1 = le.fit_transform(Data[name])
    D1 = minmax_scale(D1)
    D1 = np.round(D1,3)
    categorical_transformed_features[:,i] = D1
    i = i+1

# 4- normalizing numrical features
numerical_features = Data[numerical_columns_name]
numerical_features = np.array(numerical_features)
for i in range(len(numerical_columns_name)):
    temp = numerical_features[:,i]
    temp = minmax_scale(temp)
    temp = np.round(temp,3)
    numerical_features[:,i] = temp

# create total feature vector
All_features = np.concatenate((categorical_transformed_features,numerical_features), axis= 1)
Numeric_labels = np.reshape(Numeric_labels, (num_samples,1))
All_featurs_label = np.concatenate((All_features,Numeric_labels), axis= 1)
N = np.shape(All_featurs_label)[1]
Corr = np.zeros((N,N))
for i in range(N):
    for j  in range(N):
       temp = np.corrcoef(All_featurs_label[:,i],All_featurs_label[:,j])
       Corr[i,j]  =temp [0,1]
       

all_features_name = categorical_columns_name+ numerical_columns_name
all_features_name.append('Severity')

NewData = pd.DataFrame(data = All_featurs_label, 
                        index = range(np.shape(All_features)[0]),
                        columns = all_features_name)

NewData.to_csv("All_years_Casul_Data_Normalised.csv", index=False)  

fig2 = plt.Figure(figsize=(12,6))
sns.heatmap(Corr,xticklabels=all_features_name, yticklabels= all_features_name)
plt.show()

fig3= plt.Figure(figsize=(12,6))
plt.hist(Data['Day'])
plt.ylabel('frequency')
plt.show()


fig3= plt.Figure(figsize=(12,6))
plt.hist(Data['Month'])
plt.ylabel('frequency')
plt.show()

fig3= plt.Figure(figsize=(12,6))
plt.hist(Data['Year'])
plt.ylabel('frequency')
plt.show()

fig3= plt.Figure(figsize=(12,6))
plt.hist(Data['DayNight'])
plt.ylabel('frequency')
plt.show()

fig3= plt.Figure(figsize=(12,6))
plt.hist(Data['Stats Area'])
plt.ylabel('frequency')
plt.show()


fig3= plt.Figure(figsize=(12,6))
plt.hist(Data['Weather Cond'])
plt.ylabel('frequency')
plt.show()

