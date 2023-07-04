# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:19:53 2023

@author: betsa
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


# **-fill the unknown with most frequent value in each column
def fill_missing_most_frequent(Data, col_name, missing_val, strategy_method):  
     mostfreq_imputer = SimpleImputer(missing_values=missing_val,strategy=strategy_method)
     Data[col_name] = mostfreq_imputer .fit_transform(Data[col_name].values.reshape(-1,1))
     return Data

# **- Convert time to categorical 
def convert_time_category(Data):
    
    X_value = Data['Time']
    num_samples = np.shape (X_value)[0]
    newtime = np.zeros((num_samples,2))
    i=0
    for time in X_value:
        #print(time)
        t = time.split(':')
        hour = int (t[0])
        minute = int (t[1])
        if ((hour>=6) and (minute>=30)):
           if ((hour<=9) and (minute<=30)):
               newtime[i,0] = 1 #AM peak
        if ((hour>=15) and (minute>=30)):
          if ((hour<=18) and (minute<=30)):
              newtime[i,1] = 1  # PM Peak     
               

                        
        i = i+1
        
       
    Data ['AM_Peak'] = newtime[:,0].tolist()  
    Data ['PM_Peak'] = newtime[:,1].tolist()   
      

# **- Convert Date to categorical 
def convert_Date_category(Data):
    
    X_value = Data['Date']
    newDate = []
    i=0
    for Date in X_value: 
        t = Date.split('/')
        if (t[1]=='01') or (t[1]=='1'):
            newDate.append('Jan')
        if (t[1]=='02') or (t[1] == '2'):
            newDate.append('Feb')
        if (t[1]=='03') or (t[1]=='3'):
            newDate.append('Mar')
        if (t[1]=='04') or (t[1]=='4'):
            newDate.append('Apr')
        if (t[1]=='05') or (t[1]=='5'):
            newDate.append('May')
        if (t[1]=='06') or (t[1]=='6'):
            newDate.append('Jun')
        if (t[1]=='07') or (t[1]=='7'):
            newDate.append('Jul')
        if (t[1]=='08') or (t[1]=='8'):
            newDate.append('Aug')
        if (t[1]=='09') or (t[1]=='9'):
            newDate.append('Sep')
        if (t[1]=='10'):
            newDate.append('Oct')
        if (t[1]=='11'):
            newDate.append('Nov')
        if (t[1]=='12'):
            newDate.append('Dec')    
     
    Data['Date'] = newDate     
       
                        
    
      
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 1- Read Data
Data = pd.read_csv('CrashMLB8_new_mine.csv')
print("------------Data information----------------")
print (Data.info())

All_col_names = ['CodeAccide','LGA_NAME', 'Location', 'Date','Time',
                 'Type',	
                 'DayWeek', 'Light',	'TypeInters',	'SpeedZone',
                 'AirCond', 	'VehiNum',	'PersNum',
                 'PersNum1',	'PersInj2Nu',	'PersInj3Nu', 'PersKillNu',
                 'SEVERITY',	'PersNoInjN']


# 2- Drop  unrelated colums and the duplicated column `"CrashCount"` and "PersNum is the same
Data = Data.drop(columns=["SA2C2016",	"SA2N2017",	"SA1_20",	"SA2_20",
                          "PostCode", "months",'DCA','Type'])

# 3- Drop duplicated rows
Data = Data.drop_duplicates(subset = 'CodeAccide')
print("number of samples after drop duplicate =", np.shape(Data['CodeAccide'])[0])
print ('Data set info after removing duplicates:')
print (Data.info())


# 4- fill missing values by most common
NewData = Data.copy()  
NewData = fill_missing_most_frequent(NewData, 'LGA_NAME', np.nan ,'most_frequent')
NewData = fill_missing_most_frequent(NewData, 'Light', 'Unknown','most_frequent')
NewData = fill_missing_most_frequent(NewData, 'TypeInters', 'Unknown','most_frequent')
NewData = fill_missing_most_frequent(NewData, 'SpeedZone', 777,'most_frequent')
NewData = fill_missing_most_frequent(NewData, 'SpeedZone', 888,'most_frequent')
NewData = fill_missing_most_frequent(NewData, 'SpeedZone', 999,'most_frequent')
NewData = fill_missing_most_frequent(NewData, 'AirCond', 'Not known','most_frequent')

print("-----------New Data set info--------------------")
print (NewData.info())




# 5- set time and date
convert_Date_category(NewData)
print('unique value Date:', np.unique(NewData['Date']))


convert_time_category(NewData)
NewData = NewData.drop(columns=['Time'])
print ("-------------------New dataset info after inserting time features---------------------")
print (NewData.info())

# 6- Select categorical and numeric column
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns_name = numerical_columns_selector(NewData)
categorical_columns_name = categorical_columns_selector(NewData)

# 7- ColumnTransformer 
categorical_features = NewData[categorical_columns_name[1:8]] #1 :10
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown= 'ignore')
categorical_transformed_features = onehot_encoder.fit_transform(categorical_features)
map_categorical_name=onehot_encoder.get_feature_names_out()

# 8-Create a new Dataset with new features
numerical_features = NewData[numerical_columns_name]
numerical_features = np.array(numerical_features)
labels =  NewData["SEVERITY"]   
labels = labels.tolist()
All_features = np.concatenate((categorical_transformed_features,numerical_features), axis= 1)
features_name= map_categorical_name.tolist() + numerical_columns_name 
NewData1 = pd.DataFrame(data = All_features, 
                        index = range(np.shape(All_features)[0]),
                        columns = features_name)

NewData1['SEVERITY'] = labels
temp = NewData['CodeAccide'].tolist()
NewData1['CodeAccide'] = temp
print ("========================Created Dataset info=======================================")    
print(NewData1.info())

# 9-Write in the csv file
NewData1.to_csv('New_Dataset2.csv', index = False)


    
