# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import random

import matplotlib.pyplot as plt
#import seaborn

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer



#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# **-fill the unknown with most frequent value in each column
def fill_missing_most_frequent(Data, col_name, missing_val, strategy_method):  
     mostfreq_imputer = SimpleImputer(missing_values=missing_val,
                                      strategy=strategy_method)
    
     num = np.shape(Data[col_name])
     X = np.reshape(Data[col_name].values, (num[0],1))
     temp = mostfreq_imputer.fit_transform(X)
     for i in range(len(temp)):
         Data[col_name].iloc[i] = temp[i]
     #Data[col_name] =temp.tolist ()#temp.tolist()
     
     #print("here:,",Data[col_name])
     return Data
#--------------------------------------------------------------------
#--------------------------------------------------------------------
def fill_all_missing(Data):
    # fill in missing values by the most common
    NewData = Data.copy()  
    #print("hi")
    NewData = fill_missing_most_frequent(NewData, 'Moisture Cond',
                                         'Unknown', 'most_frequent')
    #print("hi2")
    NewData = fill_missing_most_frequent(NewData, 'Weather Cond',
                                         'Unknown','most_frequent')
    
    NewData = fill_missing_most_frequent(NewData, 'Vertical Align',
                                         'Unknown','most_frequent')
    
    NewData = fill_missing_most_frequent(NewData, 'Horizontal Align',
                                         'Unknown','most_frequent')

    #NewData = fill_missing_most_frequent(NewData, 'Veh Year','XXXX',
     #                                'most_frequent')
    
    NewData = fill_missing_most_frequent(NewData, 'Sex', 'Unknown',
                                     'most_frequent')
    
    NewData = fill_missing_most_frequent(NewData, 'Age', 'XXX',
                                     'most_frequent')
    
    #NewData = fill_missing_most_frequent(NewData, 'Licence Type', 
     #                                    'Unknown','most_frequent')
    
    NewData = fill_missing_most_frequent(NewData, 'Unit Movement',
                                     'Unknown','most_frequent')
    
    #NewData = fill_missing_most_frequent(NewData, 'Number Occupants',
     #                                'XXX','most_frequent')
    
    return NewData

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# **- Convert time to categorical 
def convert_time_category(Data):
    # Time in the form of 4:30 PM
    X_value = Data['Time']
    num_samples = np.shape (X_value)[0]
    newtime = np.zeros((num_samples,2))
    i=0
    for time in X_value:
        #print(time)
        t = time.split(':')
        AM_PM = t[3]
        #print(AM_PM)
        ###t = t[0].split(':')
        hour = int (t[0])
        minute = int (t[1])
        #print(hour, minute)
        
        if AM_PM == 'AM':
          if ((hour>=6) and (minute>=30)):
            #print(hour, minute, newtime[i,0])
            if ((hour<=9) and (minute<=30)):
                newtime[i,0] = 1 #AM peak
                #print(hour, minute, newtime[i,0])
                
        if AM_PM == 'PM':         
         if ((hour>=3) and (minute>=30)):
           if ((hour<=6) and (minute<=30)):
              newtime[i,1] = 1  # PM Peak     
               
        i = i+1
        
    Data ['AM_Peak'] = newtime[:,0].tolist()  
    Data ['PM_Peak'] = newtime[:,1].tolist()   
      
#---------------------------------------------------------
#---------------------------------------------------------
def read_crash(year):
    crash_file_name = str(year) +'_DATA_SA_Crash.csv'
    #print('File name',crash_file_name )
    crash_columns =['REPORT_ID','Stats Area','Suburb',
                            'Year',	'Month','Day',	'Time',	
                            'Area Speed',	'Position Type', 
                            'Horizontal Align',	'Vertical Align',
                            'Moisture Cond',	'Weather Cond',	'DayNight',
                            'CSEF Severity','Traffic Ctrls',
                            'ACCLOC_X',	'ACCLOC_Y']
    
    Data_crash = pd.read_csv(crash_file_name)
    NewData = Data_crash[crash_columns]
    temp = Data_crash['Entity Code']
    index = np.where (temp == 'Pedestrian')
    
    return NewData.iloc[index],crash_columns
#------------------------------------------------------
#------------------------------------------------------
def read_casualty(year):
    casul_file_name = str(year) +'_DATA_SA_Casualty.csv'
    #print('File name',casul_file_name )
    casual_columns =['REPORT_ID','Casualty Type']
    
    Data_casual = pd.read_csv(casul_file_name)
    NewData = Data_casual[casual_columns]
    casual_columns =['Casualty Type']
    
    return NewData,casual_columns
#-------------------------------------------------------------
#--------------------------------------------------------
def merge_crash_casual(Data_crash, Data_casual,crash_columns, casual_columns):
    ids_in_crash =Data_crash['REPORT_ID']
    ids_in_casual = Data_casual['REPORT_ID']
    temp_casual = Data_casual['Casualty Type']
    index_in_casual=[]
    index_in_crash=[]
    
    j =0
    for id in ids_in_crash:
        # flag = False
        index = np.where (ids_in_casual == id)
        index= np.array(index)
        num_ind = np.shape(index) [1] 
        if num_ind !=0:
            for i in range(num_ind):
              index_temp =index[0,i]
              if (temp_casual[index_temp] =='Pedestrian'):
                  index_in_casual.append(index_temp)
                  index_in_crash.append(j)
                  #print("id in crash =", id)
                  #print("index in units =",index_temp)
                  Flage = True
                  break
              #if Flage == False :
                 # print(id)
                
          
        #else:
          #print(id)
          
          
        j = j + 1   
    
    # select record from casual
    #print(np.shape(index_in_casual)) 
    index_in_casual = np.array(index_in_casual)
    df1= Data_casual.iloc[index_in_casual]
    df1 = df1.drop(columns =['REPORT_ID'])
    
    # select data from crash
    #print(np.shape(index_in_crash)) 
    index_in_crash = np.array(index_in_crash)
    df2= Data_crash.iloc[index_in_crash]
    
    # new dataframe
    Features_Crash = np. array(df2)  
    Features_Casual = np.array(df1)
    All_features = np.concatenate((Features_Crash,Features_Casual), 
                                  axis= 1)
    Features_name = crash_columns + casual_columns
    NewData = pd.DataFrame(data = All_features, 
                        index = range(np.shape(All_features)[0]),
                        columns = Features_name)
    
    
    return NewData,Features_name          

#-------------------------------------------------------------
def read_units(year):
    units_file_name = str(year) +'_DATA_SA_Units.csv'
    units_columns = ['REPORT_ID', 'Unit Type',
                    'Sex',	'Age',
                    'Unit Movement'] #'Veh Year' 'Licence Type' 'Number Occupants'
    Data_units = pd.read_csv(units_file_name)
    NewData = Data_units[units_columns]
    units_columns = ['Unit Type',
                    'Sex',	'Age',
                    'Unit Movement']
    return NewData, units_columns
#--------------------------------------------------------
def merge_crash_units(Data_crash, Data_units,crash_columns, units_columns ):
    ids_in_crash =Data_crash['REPORT_ID']
    ids_in_unts = Data_units['REPORT_ID']
    index_in_units=[]
    index_in_crash=[]
    
    j =0
    
    for id in ids_in_crash:
        #flag = False
        index = np.where (ids_in_unts == id)
        index= np.array(index)
        num_ind = np.shape(index) [1] 
        if num_ind !=0 : 
          for i in range(num_ind):
             #print(i)
             index_temp =index[0,i]
             #print(Data_units['Unit Type'][index_temp])
             if (Data_units['Unit Type'][index_temp]=='Pedestrian on Road' ) or (Data_units['Unit Type'][index_temp]=='Pedestrian on Footpath/Carpark'):
                index_in_units.append(index_temp)
                index_in_crash.append(j)
                #print("id in crash =", id)
                #print("index in units =",index_temp)
                #flag =True
                break
        #if flag== False:
         #   print(id)
        j=j+1       
             
    
    
    #print(np.shape(index_in_units)) 
    index_in_units = np.array(index_in_units)
    Data_units=Data_units.drop(columns='REPORT_ID')
    df1= Data_units.iloc[index_in_units]
    
    index_in_crash = np.array(index_in_crash)
    df2= Data_crash.iloc[index_in_crash]
    
    
    # new dataframe
    Features_Crash = np. array(df2)  
    Features_units = np.array(df1)
    All_features = np.concatenate((Features_Crash,Features_units), 
                                  axis= 1)
    Features_name = crash_columns + units_columns
    NewData = pd.DataFrame(data = All_features, 
                        index = range(np.shape(All_features)[0]),
                        columns = Features_name)
    
    return NewData  , Features_name    

#------------------------------------------------------------------------------
def convert_id_to_date(Data_crash, crash_columns, Year):
    if Year <=2020:
        Features_map,index_in_crash = look_up_id_report(Data_crash)
        Features_map = np.array(Features_map)
        #print("hi")
        Features_map = np.reshape(Features_map, newshape = (len(Features_map),1))
        
        map_columns=['exact_Date']
        index_in_crash = np.array(index_in_crash)
        df2= Data_crash.iloc[index_in_crash]
        Features_Crash = np. array(df2) 
    else:
        Features_map=random_date_id_report(Data_crash)
        Features_map = np.array(Features_map)
        Features_map = np.reshape(Features_map, newshape = (len(Features_map),1))
        
        map_columns=['exact_Date']
        Features_Crash = np. array(Data_crash) 
        
        
   
    
   # new dataframe
    
    All_features = np.concatenate((Features_Crash,Features_map), axis= 1)
    Features_name = crash_columns + map_columns
    NewData = pd.DataFrame(data = All_features, 
                            index = range(np.shape(All_features)[0]),
                            columns = Features_name)
        
    return NewData  , Features_name    

        
#-----------------------------------------------------------------------------
def look_up_id_report(Data_crash):
    Map_Data = pd.read_csv('DataSA_Crash_DateTime_2012to2021.csv')
    ids_in_crash =Data_crash['REPORT_ID']     
    ids_in_map = Map_Data['REPORT_ID']
    index_in_map=[]
    index_in_crash=[]
        
    j =0
        
    for id in ids_in_crash:
        #print("id =", id)
        index = np.where (ids_in_map == id)
        index= np.array(index)
        #print(index)
        num_ind = np.shape(index) [1] 
        if num_ind !=0 : 
            index_temp = index[0,0]
            index_in_map.append(index_temp)
            index_in_crash.append(j)
            #print("id in crash =", id)
            #print("index in units =",index_temp)
            
        j=j+1       
                 
        
        
    print(np.shape(index_in_map)) 
    index_in_map = np.array(index_in_map)
    Date= Map_Data['DATE_TIME'].iloc[index_in_map]
   
    
    Features_map=[]
    for d in Date:
        #print(d)
        t = d.split(' ')
        t[0] = t[0].replace('/','-')
        #print(t[0])
        Features_map.append(t[0])
        #break
      
    return  Features_map,index_in_crash
#------------------------------------------------------------------------------
def convert_month_2_num(long_month_name):
    datetime_object = datetime.datetime.strptime(long_month_name, "%B")
    month_number = datetime_object.month
    return month_number
#------------------------------------------------------------------------------
def random_date_id_report(Data_crash):
    ids_in_crash =Data_crash['REPORT_ID']     
    year =Data_crash['Year']	
    month=Data_crash['Month']
    day= Data_crash['Day']
    Feature_map=[]
    for y,m,d in zip(year,month, day):
        #print(y,m,d)
        m = convert_month_2_num(m)
        #print(y,m,d)
        date=find_date(y, m, d)
        Feature_map.append(date)
    return Feature_map
#------------------------------------------------------------------------------
def find_date(year, month, weekday):
    # Get start with random day of the month
    rand_day = random.randint(1,22)
    #print(rand_day)
    date = datetime.date(year, month, rand_day)
    # Find the first occurrence of the specified weekday in the month
    while date.strftime('%A') != weekday:
        date += datetime.timedelta(days=1)

    return date
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
first_year =2012
Year = first_year
for i in range(11):   
  Year = first_year + i  
  print("i =",  i, "Year =", Year)
  print ("Preparing Year:", Year)  
  Data_crash, crash_columns= read_crash(Year)    
  #print("------------------Real Crash Data info----------------------")
  #print(Data_crash.info())
  
  Data_casual, casual_columns =read_casualty(Year)
  #print("------------------Real Casualty Data info----------------------")
  #print(Data_casual.info())
  
  # merge crash and casual
  New_Data1, All_features_name = merge_crash_casual(Data_crash, Data_casual,
                                                  crash_columns,casual_columns)
  #print("--------------Merge Data Crash and Casual-----------------")
  #print(New_Data1.info())
  
  Data_units, units_columns = read_units(Year)
  #print(Data_units.info())
  New_Data, All_features_name = merge_crash_units(New_Data1, Data_units,
                                                  All_features_name,units_columns)
  #print("------------------Merge Data Crash and Units--------------------")
  #print(New_Data.info())
  #print("-----------------Map id to Date------------------- ")
  New_Data, All_features_name =convert_id_to_date(New_Data, All_features_name, Year)
  #print(New_Data.info())
  
  New_Data = fill_all_missing(New_Data)

  convert_time_category(New_Data)
  New_Data = New_Data.drop(columns=['Time'])
  if i== 0:
      Total_Data = New_Data
  else :
      Total_Data = pd.concat([Total_Data, New_Data])
    
  #print("-------------------------------------")  
  #New_Data.to_csv('example.csv', index = False)
  
  

print ("-----------------New dataset info convert time feature-------------")
print (Total_Data.info())
# 9-Write in the CSV file
Total_Data.to_csv('All_years_Casul_Data.csv', index = False)

# covert year into one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown= 'ignore')
Temp = Total_Data['Year']
Temp = np.reshape(Temp,newshape=(len(Temp),1))
categorical_transformed_features = onehot_encoder.fit_transform(Temp)
map_categorical_name=onehot_encoder.get_feature_names_out()

Total_Data_binary_Years = Total_Data.copy()
#Total_Data_binary_Years=Total_Data_binary_Years.drop(columns =['Year'])
for i in range(11):
   Total_Data_binary_Years[map_categorical_name[i]] = categorical_transformed_features[:,i].tolist()
   
 
Total_Data_binary_Years.to_csv("All_years_Casul_Data_binaryYear.csv", index=False)  
