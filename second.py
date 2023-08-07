# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:06:15 2023

@author: betsa
"""

# this function consider different combination of bins

# Standard package
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import random
import math
from itertools import combinations

#Sklearn
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
import xgboost as xg

#imabalance
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler

#tensorflow
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.layers import Layer



#------------------------------------------------------------------------------
def Prepare_data():
    Data = pd.read_csv('New_Dataset6.csv')
    '''# remove duplicated
    numerical_selector = selector(dtype_exclude = object)
    numerical_feature_names = numerical_selector(Data)
    dfObj = pd.DataFrame(Data)
    print("hi")
    new = dfObj[dfObj.duplicated(keep='last',
                                 subset = ['REGION_NAM', 
                                           'DEG_URBAN_', 'DATE',
                                           'Day_Week_D',
                                           'Light_Co_1', 
                                           'Road_Geo_1', 
                                           'Lat', 
                                           'Long',
                                           'SPEED_ZONE', 
                                           'AM_Peak', 'PM_Peak'])]
    print("new")
    print(new.info())
    Data= new'''
    

    # select column Features 
    Label_name ='Risk'
    Y = Data[Label_name]
    Y = np.array(Y)
    #print(np.unique(Y))
    #Y = np.round(Y,2)
    #print(np.unique(Y))
    # plot histogram Y
    # Creating histogram
    fig = plt.figure(figsize= (12,6))
    bins =[0,0.1, 0.2,0.3, 0.4, 0.50,0.6, 0.7,0.8,0.9, 1]
    plt.hist(Y, bins = bins)
    plt.show()
    
    mu = np.mean(Y)
    var = np.var(Y)
    for i in range(len(Y)):
        if ((Y[i]>=0.5 and Y[i]<0.6) or (Y[i]==1)):
           Y[i] = math.exp(-((Y[i]-mu)**2)/var)#math.exp(-Y[i])#
           
        else:
           Y[i] = math.exp(-((Y[i]-mu)**2)/var)#math.exp(-Y[i] )'''
           
           
           
    '''# change distribution
    Y_binned, count_Bins, Bin_indexes=bining(0, Y,10)
    s =0
    for i in range(10):
        index = Bin_indexes[i]
        s = s + np.shape(index)[1]
        Y[index] = 1+ (s/len(Y))
        #print(s/len(Y))
    '''
    fig1 = plt.figure(figsize= (12,6))
    plt.hist(Y, bins)#hist(Y)
    plt.show()       
    '''print(np.unique(Y))  '''     
    
        

    # select numercal features
    Data=Data.drop(columns=[Label_name, 'SEVERITY','DCA_Descri'])
    
    numerical_selector = selector(dtype_exclude = object)
    numerical_feature_names = numerical_selector(Data)
    num_cols = len(numerical_feature_names)
    
    ID =Data['ACCIDENT_N']
    X = Data[numerical_feature_names]
    X=np.array(X)
    
    
    
    #X = X**2
    num_features = len(numerical_feature_names)
    print("Orginal dataset shape:", np.shape(X))
    
    return X, Y,ID
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def handle_duplicate(X,Y,ID):
    # handle duplicate records with the same feature but different target
    NewX=np.zeros(np.shape(X))
    NewY=np.zeros(np.shape(Y))
    list_index=np.zeros(shape=(1,len(Y)))
    l=0
    low=0
    for i in range(250): #len(Y)):
        if (i in list_index) and (i!=0):
            #print(i)
            continue
        else :
            dist=np.sqrt(np.sum((X[i] - X) ** 2, axis=1))
            index = np.where(dist ==0.0)
            print(index)
            Y_avg=np.mean(Y[index])
            num=np.shape(index)[1]
            NewX[l,:]=X[i,:]
            NewY[l]=Y_avg
            list_index[0,low:low+num]=np.array(index)
            low=low+num
            l=l+1
            
            
    X= NewX[0:l,:]
    Y=NewY[0:l]    
    print("New dataset shape:", np.shape(X))
    return X,Y
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def bining(X_Train, Y_Train,num_Bins):
    #print("input shape bin",np.shape(X_Train))
    #print("input Y shape bin",np.shape(Y_Train))
    min_v = np.min(Y_Train)
    max_v = np.max(Y_Train)
    #print(min_v,max_v)
    bins = np.linspace(min_v, max_v,num_Bins,endpoint=False)
    Y_binned = np.digitize(Y_Train, bins)
    bin_size =(max_v -min_v)/num_Bins
    #print(bin_size)
    for i in range(num_Bins-1):
        if i==0 :
            t = Y_Train <(i+1)*bin_size
            Y_binned[t]= i+1
        else :
            t1 = Y_Train >=(i*bin_size)
            t2 = Y_Train <(i+1)*bin_size
            t = t1 & t2
            Y_binned[t]= i+1
    
    
    #print("Y_Binned shape bin",np.shape(Y_binned))
    
    # count of bins
    count_Bins= []
    Bin_indexes = []
    avg_bin = []
    min_bin =[]
    max_bin =[]
    for i in range(num_Bins):
         index = np.where(Y_binned== i+1)
         Bin_indexes.append(index)
         if np.shape(index)[1] == 0:
             count_Bins.append(0)
             avg_bin.append(0)
             max_bin.append(0)
             min_bin.append(0)
             continue
             
         count_Bins.append(np.shape(index)[1])
         avg_bin.append(np.mean(Y_Train[index]))
         max_bin.append(np.max(Y_Train[index]))
         min_bin.append(np.min(Y_Train[index]))
         #plt.figure(figsize= (12,6))
         #plt.plot(Y_Train[index],'b.')
         #plt.xlabel ("bin" + str(i)+ "samples")
         
         
   
    count_Bins = np.array(count_Bins)
    avg_bin = np.array(avg_bin)
    min_bin = np.array(min_bin)
    max_bin = np.array(max_bin)
    
    #print('avg bin=',avg_bin)
    print ("count of Bins=", count_Bins)
    #for i in range((num_Bins)):
     #   print("bin:", i, "[", min_bin[i], ",",max_bin[i], "]")
   
    
    '''plt.figure(figsize= (12,6))
    plt.plot(count_Bins)
    plt.xlabel ("bins")'''
         
    return Y_binned, count_Bins, Bin_indexes
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Create_balanced_UnderSample_dataset(X_Train,Y_Train,Y_bin):
    #Under samplin approach based on classification
    # first 
    #rus = NearMiss(version = 1, n_neighbors=1)    
    # second Create an undersampler object
    #rus =  TomekLinks()
    #Third
    #rus = EditedNearestNeighbours(n_neighbors=5)
    #fourth
    rus= RandomUnderSampler(sampling_strategy='auto')
    #print(len(Y_bin))
    #print(np.shape(X_Train))
    X_Train_Blanc, Y_bin_Blanc = rus.fit_resample(X_Train, Y_bin)
    
    index = rus.sample_indices_
    Y_Train_Blanc = np.zeros(shape=len(index))
    for i in range(len(index)):
           Y_Train_Blanc[i] = Y_Train[index[i]]
    
    #print(Y_Train_Blanc[0:30])
    #print(Y_bin_Blanc[0:30])       
           
    
    return  X_Train_Blanc , Y_Train_Blanc , Y_bin_Blanc
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Create_balanced_OverSample_dataset(X_Train,Y_Train,Y_bin):
    
    #Over sampling approach based on classification:
    
    ros = RandomOverSampler(sampling_strategy='auto')
    X_Train_Blanc, Y_bin_Blanc = ros.fit_resample(X_Train, Y_bin)
    
    index = ros.sample_indices_
    Y_Train_Blanc = np.zeros(shape=len(index))
    for i in range(len(index)):
           Y_Train_Blanc[i] = Y_Train[index[i]]
    
    return  X_Train_Blanc , Y_Train_Blanc , Y_bin_Blanc
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def create_Blanced_undersample_dataset(X_Train ,Y_Train, num_Bins):
   Y_bin, count_Bins, Bin_indexes = bining(X_Train, Y_Train,num_Bins)
   X_Train_Blanc,Y_Train_Blanc,Y_bin_Blanc=Create_balanced_UnderSample_dataset(X_Train,
                                                                               Y_Train,
                                                                               Y_bin)
   print("number of under samples:", len(Y_Train_Blanc))
   
   return X_Train_Blanc,Y_Train_Blanc
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def create_Blanced_oversample_dataset(X_Train ,Y_Train, num_Bins):
   Y_bin, count_Bins, Bin_indexes= bining(X_Train, Y_Train,num_Bins)
   X_Train_Blanc,Y_Train_Blanc,Y_bin_Blanc=Create_balanced_OverSample_dataset(X_Train,
                                                                               Y_Train,
                                                                               Y_bin)
   print("number of over samples:", len(Y_Train_Blanc))
   
   return X_Train_Blanc,Y_Train_Blanc

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def CV(X,Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size=0.3) 
    #print("CV")
    # Train
    All_UnderSample_models,All_OverSample_models,AGG_model, RMSE_US,RMSE_OS=Train_Stacking_model(X_Train,
                                                                                Y_Train)
    # Test
    Test_Stacking_model(X_Test,Y_Test,All_UnderSample_models,
                        All_OverSample_models,AGG_model,RMSE_US,RMSE_OS)
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Train_Stacking_model(X_Train,Y_Train):
    # Train Undersample models
    if Model_Option == 'Unersample':
        All_UnderSample_models,pred_US_Train,RMSE_US,Y_Train_Blanc=Train_all_Base_learners(X_Train,
                                                                             Y_Train,
                                                                             'Unersample')
   
        AGG_model,RMSE_Train_ALL= Train_Meta_learner(pred_US_Train,Y_Train)
        
        print("================UNderSampling==============")
        print("RMSE Train =", RMSE_US)  
        print("Average Train Base Learners =", np.mean(RMSE_US))
        print("===============STacking===================")
        print("==========================================")
        print("RMSE ALL Train =", RMSE_Train_ALL)
        
        return All_UnderSample_models, 0, AGG_model, RMSE_US,0
        
    if Model_Option == 'Oversample' :
    # Train Oversampling models
         All_OverSample_models,pred_OS_Train,RMSE_OS,Y_Train_Blanc=Train_all_Base_learners(X_Train,
                                                                             Y_Train,
                                                                             'Oversample')
         AGG_model,RMSE_Train_ALL= Train_Meta_learner(pred_OS_Train,Y_Train)
         print("==============OverSampling=================")
         print("RMSE Train =", RMSE_OS)  
         print("Average Base Learners =", np.mean(RMSE_OS))
         print("===============STacking===================")
         print("==========================================")
         print("RMSE ALL Train =", RMSE_Train_ALL)
         return 0, All_OverSample_models, AGG_model,0,RMSE_OS
        
        
    if Model_Option == 'Both':
        print("both")
        All_UnderSample_models,pred_US_Train,RMSE_US,Y_Train_Blanc=Train_all_Base_learners(X_Train,
                                                                             Y_Train,
                                                                             'Unersample')
        All_OverSample_models,pred_OS_Train,RMSE_OS,Y_Train_Blanc=Train_all_Base_learners(X_Train,
                                                                            Y_Train,
                                                                            'Oversample')
        
        # Aggrigate
        X= np.concatenate((pred_OS_Train,pred_US_Train), axis =1 )
        print(np.shape(X))
        AGG_model,RMSE_Train_ALL= Train_Meta_learner(X,Y_Train)
        print("========================UNderSampling=========================")
        print("RMSE Train =", RMSE_US)  
        print("Average Base Learners =", np.mean(RMSE_US))
        print("========================OverSampling=========================")
        print("RMSE Train =", RMSE_OS)  
        print("Average Base Learners =", np.mean(RMSE_OS))
        print("===============STacking===================")
        print("================================================================")
        print("RMSE ALL Train =", RMSE_Train_ALL)
        return All_UnderSample_models, All_OverSample_models, AGG_model,RMSE_US,RMSE_OS

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Train_all_Base_learners(X_Train,Y_Train,Option_Blanc):
    # Parameters
    num_features = np.shape(X_Train)[1]
    RMSE_Train =np.zeros(shape=(num_Models))
    All_models=[]
    #print("Num models:",num_Models)
    # Blanced Dataset
    if (Option_Blanc =='Unersample'):
       X_Train_Blanc,Y_Train_Blanc=create_Blanced_undersample_dataset(X_Train,
                                                                     Y_Train,
                                                                     num_Bins)
    
      
    else:
      X_Train_Blanc,Y_Train_Blanc=create_Blanced_oversample_dataset(X_Train,
                                                                       Y_Train,
                                                                       num_Bins)  
        
   
    pred_Train = np.zeros(shape= (len(Y_Train),num_Models))
    #print("shape X_Train_Balance", np.shape(X_Train_Blanc))
    l = 0
    t1,t2, Bin_indexes = bining(X_Train_Blanc,Y_Train_Blanc,num_Bins)
    #print("Train_all_base")
    for i in range(len(all_permutation)):
        temp=all_permutation[i]
        for j in range(len(temp)):
           X_Bin, Y_Bin = Select_Bin_data(X_Train_Blanc,Y_Train_Blanc,Bin_indexes,temp[j])
           RMSE_Train[l], pred_Train[:,l], model = Train_base_learner(X_Bin,Y_Bin,
                                                                X_Train,
                                                                Y_Train)
           
           #pred_Train[:,l]= pred_Train[:,l]*(1-RMSE_Train[l])
           print(RMSE_Train)
           All_models.append(model)
           l=l+1
      
    
    print("Train all Base learners")      
    return  All_models, pred_Train, RMSE_Train,Y_Train_Blanc       

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def  Select_random_data(X,Y,Bin_indexes,selected_Bin_array):
    print("select samples randomly")
    num_all_train = len(Y)
    num_train_step = np.random.randint(low =int (0.5*num_all_train), 
                                       high= num_all_train, size=1)
    print("numbe of train data for this model:", num_train_step)
    indexes = np.random.randint(low =0, high=num_all_train-1, 
                              size =num_train_step)
    
    print(np.shape(indexes))
    Y_Bin = Y[indexes] 
    X_Bin = X[indexes,:]
      
    print("X_bin shape", np.shape(X_Bin),"Y_bin shape", np.shape(Y_Bin))
    
    return X_Bin, Y_Bin 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def  Select_Bin_data(X,Y,Bin_indexes,selected_Bin_array):
    print("select",selected_Bin_array)
    
    indexes =[]
    for i in range(len(selected_Bin_array)):
        current_bin = selected_Bin_array[i]-1
        #print("current bin", current_bin)
        x = Bin_indexes[current_bin]
        #print("selectd bin index size", np.shape(x))
        indexes.append(x)
        
    indexes = np.array(indexes)
    #print("shape indexes:", np.shape(indexes))
    indexes = np.resize(indexes, new_shape=(1,np.shape(indexes)[0]*np.shape(indexes)[2]))
    #print("for train bins shape", np.shape(indexes))
    #print(np.shape(X))
    #print(np.max(indexes))
    Y_Bin = Y[indexes[0]] 
    X_Bin = X[indexes[0],:]
      
    #print("X_bin shape", np.shape(X_Bin),"Y_bin shape", np.shape(Y_Bin))
    
    return X_Bin, Y_Bin 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Train_base_learner(X_Train,Y_Train,X_Test, Y_Test):
     print("TRain_base")
     #print("shape X test", np.shape(X_Test))
     #print("shape y test", np.shape(Y_Test))
    # Train a Base learner for regression
     CL_model = Reg_model_create(Option_Base_reg,model_info_Base)
     #print("Base model", CL_model)
     num_features = np.shape(X_Train)[1]
     CL_model, W =fit_Reg_model_feature_weights(CL_model,X_Train,Y_Train,
                                                 Option_Base_reg,num_features)
     
     RMSE_Train, pred_Train =Test_Reg_model(CL_model,X_Test, Y_Test, 
                                          Option_Base_reg,num_features)
     print("end of Train base")
     
     return RMSE_Train, pred_Train, CL_model          

    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Reg_model_create(option,model_info):
    if option=='MLP':
       model = MLP_reg(model_info)
    
    #if option == 'CNN':
        #model = CNN_reg(model_info)
     
         
    if option == 'DTree':
         model = tree.DecisionTreeRegressor(criterion= model_info['criterion'], 
                                            max_depth=model_info['max_depth'])
         
         
    if option =='RF':
        model =RandomForestRegressor(n_estimators=model_info['n_estimators'],
                                     criterion=model_info['criterion'],
                                         max_depth=model_info['max_depth'])
        
    if option == 'Adaboost':
        base = tree.DecisionTreeRegressor(criterion=model_info['criterion'],
                                          max_depth=model_info['max_depth'])
        model = AdaBoostRegressor(n_estimators=model_info['n_estimators'], base_estimator =base)
    
    if option == 'xgboost':
        model = xg.XGBRegressor(objective =model_info['criterion'],
                                max_depth=model_info['max_depth'],
                                n_estimators = model_info['n_estimators'],
                                seed = 123)
    
  
    return model

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Test_Reg_model(model,X_Test, Y_Test, option, input_dim):
          
    if option == 'MLP':
        X_Test = np.reshape(X_Test, newshape=(len(Y_Test),input_dim,1))
        

    prediction=model.predict(X_Test)
    prediction = np.reshape(prediction, np.shape(Y_Test))
    RMSE= math.sqrt(np.mean((prediction-Y_Test)**2))
    #print(prediction)
             
   
    return RMSE, prediction   
   
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_Reg_model_feature_weights(model,X_Train,Y_Train, option,model_info):
    Weights = []
    if option =='MLP' or option == 'CNN':
        # fit
        print("fit reg model")
        print(np.shape(X_Train))
        model = deep_fit(model, X_Train, Y_Train,model_info["input_dim"])
        print("end of fit reg")
        # weights of Deep
        #W = model.weights[0][:]
        #for i in range(np.shape(W)[0]):  # average weights of layer 1
         #   Weights.append(np.mean(W[i])) 
        
              
    if (option == 'DTree' or option == 'RF' or option == 'xgboost' or option =='Adaboost'):
         print("fit reg model")
         print("shape x,y", np.shape(X_Train), np.shape(Y_Train))
         model =model.fit(X_Train, Y_Train)
        
         Weights = model.feature_importances_
         print("end of fit reg")
         
    if (option == 'KNN'):
        model = model.fit(X_Train, Y_Train)
         
            
    return model, Weights

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
def Train_Meta_learner(X_Train, Y_Train):
    # Aggrigate models
    AGG_model = Reg_model_create(Meta_Learner,model_info_AG)
    AGG_model, W =fit_Reg_model_feature_weights(AGG_model,X_Train,Y_Train,
                                              Meta_Learner,model_info_AG)
    RMSE_Train_ALL, pred_Train_ALL =Test_Reg_model(AGG_model,X_Train,Y_Train, 
                                       Meta_Learner,model_info_AG["input_dim"])

    
    
    return AGG_model, RMSE_Train_ALL

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Test_Stacking_model(X_Test, Y_Test,All_UnderSample_models,All_OverSample_models,AGG_model,RMSE_US,RMSE_OS):
   
  
   # Undersampling Test
   if Model_Option == 'Unersample':
       #print ("hi") 
       #print(np.shape(X_Test))
       #print(np.shape(Y_Test))
       Y_US_Pred, RMSE_US_Test=Test_all_Base_learners(X_Test,Y_Test,
                                               All_UnderSample_models)
       #for i in range(len(All_UnderSample_models)):
          #Y_US_Pred[:,i]= Y_US_Pred[:,i]* (1-RMSE_US[i])
          
       #print("end of testing base learners")
       RMSE_Test_All, Y_Pred =Test_Reg_model(AGG_model,Y_US_Pred,Y_Test, 
                                             Meta_Learner,np.shape(Y_US_Pred)[1])
       
       print("==================Result of Undersampling=========================")
       print("RMSE Test =", RMSE_US_Test)
       print("Average Base Learners =", np.mean(RMSE_US_Test))
       print("==================Result of Stacking ============================")
       print("RMSE Stacking Model Test= ",RMSE_Test_All)
       
   if Model_Option == 'Oversample' :
       Y_OS_Pred, RMSE_OS_Test=Test_all_Base_learners(X_Test,Y_Test,
                                                      All_OverSample_models)
       #for i in range(len(All_OverSample_models)):
          #Y_OS_Pred[:,i]= Y_OS_Pred[:,i]* (1-RMSE_OS[i])
    
       RMSE_Test_All, Y_Pred =Test_Reg_model(AGG_model,Y_OS_Pred,Y_Test, 
                                             Meta_Learner,np.shape(Y_OS_Pred)[1])
       print("==================Result of Oversampling=========================")
       print("RMSE Test =", RMSE_OS_Test)
       print("Average Base Learners =", np.mean(RMSE_OS_Test))
    
       print("==================Result of Stacking ============================")
       print("RMSE Stacking Model Test= ",RMSE_Test_All)
    
    
   if Model_Option == 'Both':   
    # Test Aggrigate
       Y_US_Pred, RMSE_US_Test=Test_all_Base_learners(X_Test,Y_Test,
                                                      All_UnderSample_models)
       #for i in range(len(All_UnderSample_models)):
          #Y_US_Pred[:,i]= Y_US_Pred[:,i]* (1-RMSE_US[i])
          
       Y_OS_Pred, RMSE_OS_Test=Test_all_Base_learners(X_Test,Y_Test,
                                                   All_OverSample_models)
       #for i in range(len(All_OverSample_models)):
          #Y_OS_Pred[:,i]= Y_OS_Pred[:,i]* (1-RMSE_OS[i])
    
       X = np.concatenate((Y_OS_Pred,Y_US_Pred), axis =1 )
       RMSE_Test_All, Y_Pred =Test_Reg_model(AGG_model,X,Y_Test, Meta_Learner,
                                             np.shape(X)[1])
    
       print("==================Result of Undersampling=========================")
       print("RMSE Test =", RMSE_US_Test)
       print("Average Base Learners =", np.mean(RMSE_US_Test))
       print("==================Result of Oversampling=========================")
       print("RMSE Test =", RMSE_OS_Test)
       print("Average Base Learners =", np.mean(RMSE_OS_Test))
    
       print("==================Result of Stacking ============================")
       print("RMSE Stacking Model Test= ",RMSE_Test_All)
       
   plot_predict(Y_Test, Y_Pred)    
   
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Test_all_Base_learners(X_Test,Y_Test,All_models):
    
    num_features = np.shape(X_Test)[1]
    num_models = np.shape(All_models)[0]
   
    #Test Undersample models
    pred_Test= np.zeros(shape= (len(Y_Test),num_models))
    RMSE_Test=np.zeros(shape=(num_models))
    
    for i in range(num_models):
        model =All_models[i]
        RMSE_Test[i], pred_Test[:,i] =Test_Reg_model(model,X_Test,Y_Test,
                                                     Option_Base_reg,
                                                     num_features)
    
    
          
    return pred_Test , RMSE_Test      


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def deep_fit(model,X_Train,Y_Train,input_dim):
   
    #print("x shape", np.shape(X_Train))
    #print("Y shape", np.shape(Y_Train))
    #print(input_dim)
    X_Train = np.reshape(X_Train, newshape=(len(Y_Train),input_dim,1))
    
    hist = model.fit(X_Train, Y_Train, epochs =5, verbose = 1,batch_size = 64)
    hist = hist.history
    fig = plt.figure(figsize= (12,6))
    plt.plot(hist['root_mean_squared_error'], lw=3)
    plt.ylabel('Epoch', size=15)
    plt.legend(['train RMSE'], fontsize=14)
    plt.show()
    
    return model

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def MLP_reg(model_info):
    # model create
    num_units = model_info['numnodes']
    model = tf.keras.models.Sequential()
    model.add (tf.keras.layers.Dense(input_dim = model_info['input_dim'], 
                                     units= num_units
                                     , activation= 'relu'))
    for i in range(model_info['num_lyer']):
        model.add(tf.keras.layers.Dense(units = num_units/(2**(i+1)), activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.2))
   
    model.add(tf.keras.layers.Dense(1,activation=None))
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=tf.keras.metrics.RootMeanSquaredError())# 'categorical_crossentropy'
    model.summary()
    return model
#------------------------------------------------------------------------------       
#------------------------------------------------------------------------------
def plot_predict(Y_Test, Y_Pred):
    # Plt the Test Result 
    plt.figure(figsize= (12,6))
    max_value_Y = np.max(Y_Test)
    min_value_Y = np.min(Y_Test)
    t=[min_value_Y,max_value_Y]
    plt.plot(t,t,'r:')
    plt.xlim(min_value_Y,max_value_Y)
    plt.ylim(min_value_Y,max_value_Y)

    plt.plot(Y_Pred,Y_Test, "b.", label="final")
    plt.xlabel('Y predicted')
    plt.ylabel("True Values")


    #-------------------------
    diff = np.abs(Y_Pred-Y_Test)
    print("avgerage diff",np.mean(diff))
    print("--------------------------------------------------")
    ind = np.where (diff>0.3)
    ind = np.array(ind)
    for i in range(20):
        ind1=int(ind[0][i])
        print("Y =",Y_Test[ind1] , "Pred =", Y_Pred[ind1]) 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#!- Read datase
X, Y ,ID= Prepare_data()

# set parameters
num_Bins= 5
Model_Option = 'Unersample'# 'Oversample'#'Both' ##'Unersample'##
Meta_Learner='MLP'
Option_Base_reg = 'xgboost'

model_info_Base={"input_dim":np.shape(X)[1],"n_estimators":150,"max_depth":4,
           'criterion': 'reg:squarederror'}#{"input_dim":np.shape(X)[1],"num_lyer":3, "numnodes":256}
#model_info_Base=set_model_info(Option_Base_reg, Pram_Base)


                             
num_Models = 0
all_permutation=[]
for i in range(1,num_Bins+1):
    x=list(combinations(range(1,num_Bins+1),i))
    all_permutation.append(np.array(x))
    num_Models = num_Models+ len(x)
    
all_permutation = np.array(all_permutation)   

model_info_AG ={"input_dim":num_Models,"num_lyer":3, "numnodes":256}
#{"input_dim":num_Models,"n_estimators":400,"max_depth":6,'criterion': 'squared_error'}
#model_info_AG=set_model_info(Meta_Learner, Pram_Meta)
 
    
# Train and test
CV(X,Y)   
    
