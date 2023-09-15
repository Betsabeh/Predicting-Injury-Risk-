# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 23:44:31 2023

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Create_numeric_features(Data):
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
    All_features = np.concatenate((categorical_transformed_features,numerical_features), 
                                  axis= 1)
    All_features_name = categorical_columns_name+ numerical_columns_name
    
    return All_features, All_features_name
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def  plot_freq(Data,Y, col_name):
    temp = Data[col_name]
    temp= np.reshape(temp, newshape=(1,1677))
    Y= np.reshape(Y, newshape=(1,1677))
    print(np.shape(Y))
    plt.rc("font", size=14)
    pd.crosstab(temp,Y).plot(kind='bar')
    plt.title('Purchase Frequency for Month')
    plt.xlabel('Month')
    plt.ylabel('Frequency of Purchase')
    #plt.savefig('pur_dayofweek_bar')
#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------
# 1- Read Data
Data = pd.read_csv('All_Years_Data_binaryYear.csv')
Data = Data.drop(columns= ['REPORT_ID.1'])
print("------------Data information----------------")
print (Data.info())
num_samples = np.shape(Data['REPORT_ID'])[0]
print("number of samples =", num_samples)
Label = Data['CSEF Severity']
Data = Data.drop(columns= ['REPORT_ID', 'Year', 'CSEF Severity'])
# convert labels into 0, 1
Labels_class = np.unique(Label)
Y =  np.zeros((len(Label),1))
ind = np.where (Label == '4: Fatal')
Y [ind] =1
#plot_freq(Data,Y, "Age")


# 2- features
X, All_features_name = Create_numeric_features(Data)

# 3- Create model
model = RandomForestClassifier(n_estimators=500)
model.fit(X, Y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v*100))


# 4-Cross validation
num_folds =5
CV = KFold(n_splits= num_folds, shuffle=True)
F =[]
ACC =[]
Pre=[]
Rec=[]
for train_index, test_index in CV.split(X,Y):
    X_train = X[train_index,:]
    Y_train = Y[train_index]
    X_test = X[test_index,:]
    Y_test = Y[test_index]
    
    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_test = np.reshape(Y_test, newshape=(len(Y_pred)))
   

    
    ACC.append(model.score(X_test, Y_test))
    Pre.append(f1_score(Y_test, Y_pred ))
    Rec.append(f1_score(Y_test, Y_pred))
    F.append(f1_score(Y_test, Y_pred, average ='binary'))
    
    

print("---------------Result of 5 folds------------------")
print("Avg F =", np.mean(F), "std =", np.std(F))   
print("Avg ACC =", np.mean(ACC), "std =", np.std(ACC))   
print("Avg Pre=", np.mean(Pre), "std =", np.std(Pre))   
print("Avg Recall =", np.mean(Rec), "std =", np.std(Rec))    
    
    
    
    
    
