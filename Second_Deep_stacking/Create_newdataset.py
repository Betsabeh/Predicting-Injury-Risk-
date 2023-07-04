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
from sklearn.preprocessing import minmax_scale
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
    
    X_value = Data['TIME']
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
    
    X_value = Data['DATE']
    newDate = []
    i=0
    for Date in X_value: 
        #print(Date)
        t = Date.split('/')
        #print(t)
        if (t[0]=='01') or (t[0]=='1'):
            newDate.append('Jan')
        if (t[0]=='02') or (t[0] == '2'):
            newDate.append('Feb')
        if (t[0]=='03') or (t[0]=='3'):
            newDate.append('Mar')
        if (t[0]=='04') or (t[0]=='4'):
            newDate.append('Apr')
        if (t[0]=='05') or (t[0]=='5'):
            newDate.append('May')
        if (t[0]=='06') or (t[0]=='6'):
            newDate.append('Jun')
        if (t[0]=='07') or (t[0]=='7'):
            newDate.append('Jul')
        if (t[0]=='08') or (t[0]=='8'):
            newDate.append('Aug')
        if (t[0]=='09') or (t[0]=='9'):
            newDate.append('Sep')
        if (t[0]=='10'):
            newDate.append('Oct')
        if (t[0]=='11'):
            newDate.append('Nov')
        if (t[0]=='12'):
            newDate.append('Dec')    
     
     
    #print(newDate)
    Data['DATE'] = newDate     
       
                        
    
      
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 1- Read Data
Data = pd.read_csv('CrashMLB2211797_withrisk.csv')
print("------------Data information----------------")
print (Data.info())

All_col_names = ['ACCIDENT_N',	'LGA_NAME_A',	'REGION_NAM',	'DEG_URBAN_',
                 'DATE',	'TIME',	'Type_Desc',	'Day_Week_D',	'DCA_Descri',
                 'Light_Co_1','SEVERITY','Risk','DCA_Descri',
                 'NO_PERSONS',	'NO_PERSO_1',	'NO_PERSO_2',	
                 'NO_PERSO_3',	'NO_PERSO_4','POLICE_ATT',	'SPEED_ZONE', 'Road_Geo_1']


# 2- Drop  unrelated columns 
Data = Data.drop(columns=['OBJECTID','NODE_ID','NODE_TYPE',	'VICGRID94_',	
                          'VICGRID941','LGA_NAME_A','Lat',	'Long',
                          'POSTCODE_N', 'TYPE','DAY_OF_WEE','DCA_CODE',
                          'DIRECTORY','EDITION','PAGE','GRID_REFER','LIGHT_COND',
                          'GRID_REF_1','POLICE_ATT',
                          'NODE_ID_1',	'NO_OF_VEHI',	'ROAD_GEOME',
                          'DIRECTORY', 'NO_PERSONS','NO_PERSO_1',
                          'NO_PERSO_2','NO_PERSO_3','NO_PERSO_4'])

# 3- Drop duplicated rows
print("number of samples before drop duplicate =", np.shape(Data['ACCIDENT_N'])[0])
Data = Data.drop_duplicates(subset = 'ACCIDENT_N')
print("number of samples after drop duplicate =", np.shape(Data['ACCIDENT_N'])[0])
print ('Data set info after removing duplicates:')
print (Data.info())

index =[]
Temp_risk = Data ['Risk']
Temp_risk =np.array(Temp_risk)
for v,i in zip(Temp_risk, range(np.shape(Temp_risk)[0])):
    if v> 1:
        Temp_risk[i] =1.0
        index.append(i)
    
Data['Risk'] = Temp_risk.tolist()        

# 4- fill in missing values by the most common
NewData = Data.copy()  
NewData = fill_missing_most_frequent(NewData, 'Light_Co_1', 'Unknown','most_frequent')
NewData = fill_missing_most_frequent(NewData, 'REGION_NAM','Unknown','most_frequent')
NewData = fill_missing_most_frequent(NewData, 'Road_Geo_1','Unknown','most_frequent')
NewData = fill_missing_most_frequent(NewData, 'LGA_NAME', 'Unknown','most_frequent')
NewData = fill_missing_most_frequent(NewData, 'SPEED_ZONE', 777,'most_frequent')
NewData = fill_missing_most_frequent(NewData, 'SPEED_ZONE', 888,'most_frequent')
NewData = fill_missing_most_frequent(NewData, 'SPEED_ZONE', 999,'most_frequent')


print("-----------New Data set info--------------------")
print (NewData.info())




# 5- set time and date
convert_Date_category(NewData)
print('unique value Date:', np.unique(NewData['DATE']))


convert_time_category(NewData)
NewData = NewData.drop(columns=['TIME'])
print ("-------------------New dataset info after inserting time features---------------------")
print (NewData.info())

# 6- Select categorical and numeric column
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns_name = numerical_columns_selector(NewData)
categorical_columns_name = categorical_columns_selector(NewData)
categorical_columns_name = categorical_columns_name[1:]

numerical_features = NewData[numerical_columns_name]
numerical_features = np.array(numerical_features)
for i in range(np.shape(numerical_features)[1]):
    temp = numerical_features[:,i]
    temp = minmax_scale(temp)
    temp = np.round(temp,3)
    numerical_features[:,i] = temp

# 7- ColumnTransformer 
num_cat_feature = len(categorical_columns_name)
num_samples = np.shape(numerical_features)[0]
categorical_transformed_features = np.zeros(shape =(num_samples,num_cat_feature))
le = LabelEncoder()
i=0
for name in categorical_columns_name:
    D1 = le.fit_transform(NewData[name])
    D1 = minmax_scale(D1)
    D1 = np.round(D1,3)
    categorical_transformed_features[:,i] = D1
    i = i+1


# 8-Create a new Dataset with new features
labels =  NewData["SEVERITY"]   
labels = labels.tolist()
All_features = np.concatenate((categorical_transformed_features,numerical_features), axis= 1)
features_name= categorical_columns_name + numerical_columns_name 
NewData1 = pd.DataFrame(data = All_features, 
                        index = range(np.shape(All_features)[0]),
                        columns = features_name)

NewData1['SEVERITY'] = labels
temp = NewData['ACCIDENT_N'].tolist()
NewData1['ACCIDENT_N'] = temp
print ("========================Created Dataset info=======================================")    
print(NewData1.info())

# 9-Write in the CSV file
NewData1.to_csv('New_Dataset5.csv', index = False)
