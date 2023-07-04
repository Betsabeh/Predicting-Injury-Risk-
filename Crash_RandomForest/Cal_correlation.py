# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:04:52 2023

@author: betsa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.compose import make_column_selector as selector
import matplotlib.pyplot as plt
import math
import seaborn as sns

#--------------------------------------------------------------------
# **-find correlation and plot
def find_correlation(Data, target_name,col_names, title_text, feature_name):
    Y = Data[target_name]
    N = len(col_names)
    Corr_mat = np.zeros(shape=(1, N))
    i=0
    
    for name in col_names:
        Corr_mat[0,i] = np.corrcoef(Data[name], Y)[0,1]
        i = i+1
                  
    
    f = plt.figure(figsize=(16, 8))
    sns.heatmap(Corr_mat, annot = True, fmt='.2g', 
                xticklabels= col_names, cmap= 'coolwarm')
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    plt.title( title_text)
    plt.xlabel(feature_name+" Feature Values", loc="center", fontdict= font)
    plt.ylabel("Pearson Correlation Values", loc= "center", fontdict= font)

    
def find_correlation1(Data,index_temp, Y, col_names, feature_name):
    
    N = len(col_names)
    Corr_mat = np.zeros(shape=(1, N))
    i=0
    for name in col_names:
        X=Data[name]
        X= np.array(X)
        X= X[index_temp]
        Corr_mat[0,i] = np.corrcoef(X, Y)[0,1]
        i = i+1
               
    print( Corr_mat)
    f = plt.figure(figsize=(16, 8))
    sns.heatmap(Corr_mat, annot = True, fmt='.2g', 
                xticklabels= col_names,  cmap= 'coolwarm')
    plt.title(feature_name +' Pearson Correlations')
    
    
def scatter_diagram(Data,col_names, feature_name):
        Y = Data['PerNum1']
        i =1
        for name in col_names:
            X = Data[name] * i
            plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
            plt.scatter(X, Y, label=name+ f' Correlation = {np.round(np.corrcoef(X,Y)[0,1], 3)}')
            i = i+1
        # Plot
        plt.title(feature_name +' Scatterplot and Correlations')
        plt.legend(loc='best')
        plt.xlim(0.5,i-0.5) # because zero values in one-hot encoding is not meaningful
        plt.show()
        
def bar_diagram(Data,col_names, feature_name):
        Y = Data['PerNum1']
        s =[]
        for name in col_names:
            X = Data[name] 
            s.append(np.sum(X))
        # Plot
        X = np.array(X)
        plt.bar(X,Y)
        plt.title(feature_name +' Frequency')
        plt.legend(loc='best')
        plt.show()

def find_in_array(a, Key):
    N= np.shape(a)[0]
    index =[]
    for i in range(N):
        if a[i] == Key:
            index.append(i)
            
     
    index = np.array(index)        
    return index 


def plot_bar(Data,index_temp, Y, col_names, feature_name):
    
    N = len(col_names)
    Count_mat = np.zeros(shape=(1, N))
    i=0
    for name in col_names:
        X=Data[name]
        X= np.array(X)
        X= X[index_temp]
        Count_mat[0,i] = np.sum(X)
        i = i+1
               
    print( Count_mat)
    return Count_mat 

def find_most_freq(col_names):
    count_arr = []
    for name in col_names:
        X= Data[name]
        count_arr.append(np.sum(X))

    count_arr = np.array(count_arr)
    print(count_arr)
    
    index=np.argsort(-1* count_arr)
    max_freq_col =[]
    for i in range(10):
        max_freq_col.append(col_names[index[i]] )
        
    
    return max_freq_col
    
#-----------------------------------------------------------------
# 1- read the dataset
Data = pd.read_csv('New_Dataset1.csv')
print(Data.info())

# 2-Drop unused columns and blank target
Data= Data.drop(columns= ['Lat',	'Long',
                    'VehiNum',	'PersNum',	'SEVERITY',	'PersInj2Nu',	
                    'PersInj3Nu',	'PersKillNu',	'PersNoInjN', 'CodeAccide'])

#Data = Data.dropna(subset=['SEVERITY'])
#print(Data.info())

# 3- Features and label
Label = 'PerNum1'#'SEVERITY'
Y = Data[Label]
Y = np.array(Y)

numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns_name = numerical_columns_selector(Data)
# 4- pearson correlation
find_correlation(Data,Label, numerical_columns_name[19:24], 
                 'Pearson Correlations between Location and PerNum1', "Location")

find_correlation(Data,Label ,numerical_columns_name[24:36],
                 'Pearson Correlations between Months and PerNum1',"Months")

find_correlation(Data,Label, numerical_columns_name[45:52], 
                 'Pearson Correlations between WeekDay and PerNum1', "WeekDay")

find_correlation(Data,Label, numerical_columns_name[132:138], 
                 'Pearson Correlations between Light and PerNum1', "Light")

find_correlation(Data,Label, numerical_columns_name[138:145], 
                 'Pearson Correlations between TypeInters and PerNum1', "TypeInters")

find_correlation(Data,Label, numerical_columns_name[146:153], 
                 'Pearson Correlations between AirCondition and PerNum1', "AirCondition")

find_correlation(Data,Label, numerical_columns_name[155:], 
                 'Pearson Correlations between Time and PerNum1', "Time")

max_freq_col=find_most_freq(numerical_columns_name[0:19])
find_correlation(Data,Label, max_freq_col, 
                 'Pearson Correlations between top 10 LGA features and PerNum1', "LGA")

max_freq_col=find_most_freq(numerical_columns_name[52:132])
find_correlation(Data,Label, max_freq_col, 
                 'Pearson Correlations between top 10 DCA features  and PerNum1', "DCA")

#- find frequency
#sns.distplot(Y)
unique_Y = np.unique(Y)
count_Y = []
temp = Y.tolist()
for i in range(len(unique_Y)):
     count_Y.append(temp.count(unique_Y[i]))


count_Y = np.array(count_Y)
plt.plot(unique_Y,count_Y)
index=np.argsort(-1* count_Y)
all_index =[]
for i in range(5):
    index_temp = find_in_array(Y, unique_Y[index[i]])
    Count_mat=plot_bar(Data,index_temp, Y,numerical_columns_name[146:153],
             'AirCondition with PerNum1='+ str(unique_Y[index[i]]))
    
    f = plt.figure(figsize=(16, 8))
    sns.heatmap(Count_mat, annot = True, fmt='.2g', 
                xticklabels= numerical_columns_name[146:153], cmap= 'coolwarm')
    plt.title('AirCondition with PerNum1='+ str(unique_Y[index[i]]) )    


'''#all_index = np.array()
#find_correlation1(Data,all_index,Y[all_index],
                  #numerical_columns_name[146:153],  'AirCondition'+ "PerNum1 with value=")
'''
'''# 4- scatter plot
#scatter_diagram(Data,numerical_columns_name[24:36], 'Months')
#scatter_diagram(Data,numerical_columns_name[45:52] , 'WeekDay')
#scatter_diagram(Data,numerical_columns_name[146:153], 'AirCondition')
'''
'''
# Bar
for i in range(7):
    find_correlation(Data,numerical_columns_name[146+i],
                 numerical_columns_name[52:132],
                 numerical_columns_name[146+i])'''

