# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:01:23 2023

@author: betsa
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error


def fit_model_plot_importance(model,X_Train, Y_Train, X_Test, Y_Test):
    clf = model.fit(X_Train, Y_Train)
    Y_hat= clf.predict(X_Test)
    print('RMSE',math.sqrt(mean_squared_error(Y_Test, Y_hat)))
    acc = 0
    for i in range(len(Y_hat)):
        acc = acc + (Y_hat[i] - Y_Test[i])**2
       # if (Y_hat[i] == Y_Test[i] ):
            #acc = acc+1
            
    
    acc = acc / len(Y_Test)
    acc = math.sqrt (acc)
    # show the imporatnce of each features
    importance = model.feature_importances_
    
    plt.plot(Y_hat,Y_Test,'.')
    t=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    plt.plot(t,t,'r:')
    plt.xlabel('Predicted label',size=10)
    plt.ylabel('True label',size=10)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    return acc , importance
 
  
def plot_importance (importance, features_name ):
    print(np.shape(importance))
    
    indices = np.argsort(-importance)
    for f in range(len(features_name)):
        print("%2d) %-*s  %f" % (f + 1, 40,features_name[indices[f]],
                                importance[indices[f]]))
    
    my_fig = plt.figure(figsize= (40,16))
    plt.title('Feature importance')
    plt.bar(range(10), importance[indices[0:10]])
    plt.xticks(range(10),features_name[indices[0:10]])
    plt.show()




# 1- read dataset
Data = pd.read_csv('New_Dataset2.csv')
print(Data.info())

# 2-Drop unused colunms and blank traget
Data= Data.drop(columns= ['Lat',	'Long',
                    'VehiNum',	'PersNum',	'SEVERITY',	'PersInj2Nu',	
                    'PersInj3Nu',	'PersKillNu',	'PersNoInjN', 'CodeAccide'])

#Data = Data.dropna(subset=['SEVERITY'])
print(Data.info())

# 3- Features and label
Label = 'PerNum1'#'SEVERITY'
Y = Data[Label]
Data= Data.drop(columns= Label)
Y = np.array(Y)

numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns_name = numerical_columns_selector(Data)
X = Data [numerical_columns_name]
X = np.array(X)

# 4-Cross-validation
KFold_CV = KFold (n_splits= 10, shuffle= True)
model = RandomForestRegressor(n_estimators = 100, criterion='squared_error', 
                                max_depth =10)

#X_Train, X_Test, Y_Train, Y_Test = train_test_split( X, Y, test_size=0.30, random_state=42)
#fit_model_plot_importance(model,X_Train, Y_Train, X_Test, Y_Test)


acc = np.zeros((1,10))
avg_importance = np.zeros(shape =(1,np.shape(X)[1] ))
i = 0
for Train_index, Test_index in KFold_CV.split(X):
    X_Train = X [Train_index,:]
    Y_Train = Y[Train_index]
    X_Test = X [Test_index,:]
    Y_Test = Y[Test_index]
    acc[0,i], importance = fit_model_plot_importance(model,X_Train, Y_Train, X_Test, Y_Test)
    avg_importance = np.add(avg_importance, importance)
    print("----------------fold =%d---------------------" %(i))
    print ("Accuracy = ", acc[0,i])
    i = i+1


avg_importance = avg_importance /10
plot_importance (importance, np.array(numerical_columns_name) )    
mean_acc = np.mean(acc)
std_acc =np.std(acc)
print('-----------------------------------------------------')  
print ("avg 10 fold acc + std acc =",mean_acc, '+', std_acc) 

