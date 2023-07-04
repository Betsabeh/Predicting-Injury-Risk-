# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:41:52 2023

@author: betsa
"""

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
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
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
import xgboost as xg
from keras.utils import np_utils
import tensorflow as tf
import random
import math
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Prepare_data():
    Data = pd.read_csv('New_Dataset5.csv')

    # select column Features 
    Label_name ='Risk'
    Y = Data[Label_name]
    Y = np.array(Y)
    #Y = Y**0.5
    #for i in range(len(Y)):
     #   Y[i]= math.exp(-1*Y[i])
    #Y = np.round(Y,2)
    #print(np.unique(Y))
    # plot histogram Y
    # Creating histogram
    fig = plt.figure(figsize= (12,6))
    plt.hist(Y, bins = [0, 0.25, 0.50, 0.75, 1])
 
        

    # select numercal features
    Data=Data.drop(columns=[Label_name, 'SEVERITY'])
    
    numerical_selector = selector(dtype_exclude = object)
    numerical_feature_names = numerical_selector(Data)
    num_cols = len(numerical_feature_names)
    
    ID =Data['ACCIDENT_N']
    X = Data[numerical_feature_names]
    X=np.array(X)
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
    for i in range(len(Y)):
        if (i in list_index) and (i!=0):
            #print(i)
            continue
        else :
            dist=np.sqrt(np.sum((X[i] - X) ** 2, axis=1))
            index = np.where(dist ==0.0)
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
    min = np.min(Y_Train)
    max = np.max(Y_Train)
    bins = np.linspace(min, max,num_Bins,endpoint=False)
    Y_binned = np.digitize(Y_Train, bins)
    
    
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
    rus =  TomekLinks()
    #Third
    #rus = EditedNearestNeighbours(n_neighbors=5)
    #fourth
    #rus= RandomUnderSampler(sampling_strategy='auto')
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
def CV(X,Y,num_Bins_base,Model_Option,Option_Base_reg,num_Models,Meta_Learner):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size=0.1) 
   
    # Train
    All_UnderSample_models,All_OverSample_models,AGG_model=Train_Stacking_model(X_Train,
                                                                                Y_Train,
                                                                                Model_Option,
                                                                                num_Bins_base,
                                                                                Option_Base_reg,
                                                                                num_Models,
                                                                                Meta_Learner)
    # Test
    Test_Stacking_model(X_Test,Y_Test,Model_Option,All_UnderSample_models,
                        All_OverSample_models,AGG_model,
                        Option_Base_reg, Meta_Learner)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Train_Stacking_model(X_Train,Y_Train,Model_Option,num_Bins_base,Option_Base_reg,num_Models,Meta_Learner):
    # Train Undersample models
    if Model_Option == 'Unersample':
        All_UnderSample_models,pred_US_Train,RMSE_US=Train_all_Base_learners(X_Train,
                                                                             Y_Train,
                                                                             num_Bins_base,
                                                                             Option_Base_reg,
                                                                             num_Models, 'Unersample')
   
        AGG_model,RMSE_Train_ALL= Train_Meta_learner(pred_US_Train,Y_Train,
                                                 Meta_Learner)
        
        print("================UNderSampling==============")
        print("RMSE Train =", RMSE_US)  
        print("Average Base Learners =", np.mean(RMSE_US))
        print("===============STacking===================")
        print("==========================================")
        print("RMSE ALL Train =", RMSE_Train_ALL)
        
        return All_UnderSample_models, 0, AGG_model
        
    if Model_Option == 'Oversample' :
    # Train Oversampling models
         All_OverSample_models,pred_OS_Train,RMSE_OS=Train_all_Base_learners(X_Train,
                                                                             Y_Train,
                                                                             num_Bins_base,
                                                                             Option_Base_reg,
                                                                             num_Models, 'Oversample')
         AGG_model,RMSE_Train_ALL= Train_Meta_learner(pred_OS_Train,Y_Train,
                                                 Meta_Learner)
         print("==============OverSampling=================")
         print("RMSE Train =", RMSE_OS)  
         print("Average Base Learners =", np.mean(RMSE_OS))
         print("===============STacking===================")
         print("==========================================")
         print("RMSE ALL Train =", RMSE_Train_ALL)
         return 0, All_OverSample_models, AGG_model
        
    if Model_Option == 'Both':
        print("both")
        All_UnderSample_models,pred_US_Train,RMSE_US=Train_all_Base_learners(X_Train,
                                                                             Y_Train,
                                                                             num_Bins_base,
                                                                             Option_Base_reg,
                                                                             num_Models, 'Unersample')
        All_OverSample_models,pred_OS_Train,RMSE_OS=Train_all_Base_learners(X_Train,
                                                                            Y_Train,
                                                                            num_Bins_base,
                                                                            Option_Base_reg,
                                                                            num_Models, 'Oversample')
        
        # Aggrigate
        X= np.concatenate((pred_OS_Train,pred_US_Train), axis =1 )
        print(np.shape(X))
        AGG_model,RMSE_Train_ALL= Train_Meta_learner(X,Y_Train,
                                                 Meta_Learner)
        print("========================UNderSampling=========================")
        print("RMSE Train =", RMSE_US)  
        print("Average Base Learners =", np.mean(RMSE_US))
        print("========================OverSampling=========================")
        print("RMSE Train =", RMSE_OS)  
        print("Average Base Learners =", np.mean(RMSE_OS))
        print("===============STacking===================")
        print("================================================================")
        print("RMSE ALL Train =", RMSE_Train_ALL)
        return All_UnderSample_models, All_OverSample_models, AGG_model


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Train_all_Base_learners(X_Train,Y_Train, num_Bins,Option_Base_reg,num_Models, Option_Blanc):
    # Parameters
    num_features = np.shape(X_Train)[1]
    pred_Train = np.zeros(shape= (len(Y_Train),num_Models))
    RMSE_Train =np.zeros(shape=(num_Models))
    All_models=[]
    print("Num models:",num_Models)
    
    # Blanced Dataset
    if (Option_Blanc =='Unersample'):
      X_Train_Blanc,Y_Train_Blanc=create_Blanced_undersample_dataset(X_Train,
                                                                       Y_Train,
                                                                       num_Bins)
      
        
    else:
        X_Train_Blanc,Y_Train_Blanc=create_Blanced_oversample_dataset(X_Train,
                                                                         Y_Train,
                                                                         num_Bins)
    
  
        
    l =0
    for i in range(num_Models):
       
       RMSE_Train[l], pred_Train[:,l], model = Train_base_learner(X_Train_Blanc,
                                                                  Y_Train_Blanc,
                                                                  X_Train,Y_Train,
                                                                  model_info,
                                                                  Option_Base_reg )
       All_models.append(model)
       l=l+1
      
          
    return  All_models, pred_Train, RMSE_Train       
           
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Train_base_learner(X_Train,Y_Train,X_Test, Y_Test,model_info,Option_Base_reg ):
    # Train a Base learner for regression
     CL_model = Reg_model_create(Option_Base_reg,model_info)
     num_features = np.shape(X_Train)[1]
     CL_model, W =fit_Reg_model_feature_weights(CL_model,X_Train,Y_Train,
                                                 Option_Base_reg,num_features)
     RMSE_Train, pred_Train =Test_Reg_model(CL_model,X_Test, Y_Test, 
                                          Option_Base_reg,num_features)
     
     return RMSE_Train, pred_Train, CL_model
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
def Train_Meta_learner(X_Train, Y_Train,Meta_Learner):
    # Aggrigate models
    AGG_model = Reg_model_create(Meta_Learner,model_info_AG)
    AGG_model, W =fit_Reg_model_feature_weights(AGG_model,X_Train,Y_Train,
                                              Meta_Learner,model_info_AG)
    RMSE_Train_ALL, pred_Train_ALL =Test_Reg_model(AGG_model,X_Train,Y_Train, 
                                       Meta_Learner,0)

    
    
    return AGG_model, RMSE_Train_ALL

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Test_Stacking_model(X_Test, Y_Test,Model_Option,All_UnderSample_models,All_OverSample_models,AGG_model,Option_Base_reg, Meta_learner):
    
    # Undersampling Test
   if Model_Option == 'Unersample':
       Y_US_Pred, RMSE_US_Test=Test_all_Base_learners(X_Test,Y_Test,Option_Base_reg,
                                               All_UnderSample_models)
       
       RMSE_Test_All, Y_Pred =Test_Reg_model(AGG_model,Y_US_Pred,Y_Test, Meta_learner,0)
       
       print("==================Result of Undersampling=========================")
       print("RMSE Test =", RMSE_US_Test)
       print("Average Base Learners =", np.mean(RMSE_US_Test))
       print("==================Result of Stacking ============================")
       print("RMSE Stacking Model Test= ",RMSE_Test_All)
       
   if Model_Option == 'Oversample' :
       Y_OS_Pred, RMSE_OS_Test=Test_all_Base_learners(X_Test,Y_Test,
                                                   Option_Base_reg,All_OverSample_models)
    
       RMSE_Test_All, Y_Pred =Test_Reg_model(AGG_model,Y_OS_Pred,Y_Test, Meta_learner,0)
       print("==================Result of Oversampling=========================")
       print("RMSE Test =", RMSE_OS_Test)
       print("Average Base Learners =", np.mean(RMSE_OS_Test))
    
       print("==================Result of Stacking ============================")
       print("RMSE Stacking Model Test= ",RMSE_Test_All)
    
    
   if Model_Option == 'Both':   
    # Test Aggrigate
       Y_US_Pred, RMSE_US_Test=Test_all_Base_learners(X_Test,Y_Test,Option_Base_reg,
                                               All_UnderSample_models)
       Y_OS_Pred, RMSE_OS_Test=Test_all_Base_learners(X_Test,Y_Test,
                                                   Option_Base_reg,All_OverSample_models)
    
       X = np.concatenate((Y_OS_Pred,Y_US_Pred), axis =1 )
       RMSE_Test_All, Y_Pred =Test_Reg_model(AGG_model,X,Y_Test, Meta_learner,0)
    
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
def Test_all_Base_learners(X_Test,Y_Test,Option_Base_reg,All_models):
    
    num_features = np.shape(X_Test)[1]
    num_models = np.shape(All_models)[0]
    #Test Undersample models
    pred_Test= np.zeros(shape= (len(Y_Test),num_models))
    RMSE_Test=np.zeros(shape=(num_models))
    

    for i in range(num_models):
        model =All_models[i]
        RMSE_Test[i], pred_Test[:,i] =Test_Reg_model(model,X_Test,Y_Test, 
                                          Option_Base_reg,num_features)
    
    
          
    return pred_Test , RMSE_Test      
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
def plot_predict(Y_Test, Y_Pred):
    # Plt the Test Result 
    plt.figure(figsize= (12,6))
    t=[0,0.1,0.2,0.5,1]
    plt.plot(t,t,'r:')
    plt.xlim(0.2,1)
    plt.ylim(0.2,1)

    plt.plot(Y_Pred,Y_Test, "b.", label="final")
    plt.xlabel('Y predicted')
    plt.ylabel("True Values")


    #-------------------------
    diff = np.abs(Y_Pred-Y_Test)
    print("avgerage diff",np.mean(diff))
    print("--------------------------------------------------")
    ind = np.where (diff>0.5)
    ind = np.array(ind)
    for i in range(20):
        ind1=int(ind[0][i])
        print("Y =",Y_Test[ind1] , "Pred =", Y_Pred[ind1]) 
        

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def Reg_model_create(option,model_info):
    if option=='MLP':
       model = MLP_reg(model_info)
    
    if option == 'CNN':
        model = CNN_reg(model_info)
     
         
    if option == 'DTree':
         model = tree.DecisionTreeRegressor(criterion='squared_error', max_depth=6)
         
         
    if option =='RF':
        model =RandomForestRegressor(n_estimators=model_info[0], criterion='squared_error',
                                     max_depth=model_info[1])
        
    if option == 'Adaboost':
        base = tree.DecisionTreeRegressor(criterion='squared_error', max_depth=4)
        model = AdaBoostRegressor(n_estimators=100, base_estimator =base)
    
    if option == 'xgboost':
        model = xg.XGBRegressor(objective ='reg:squarederror',max_depth=5,
                                n_estimators = 100, seed = 123)
    
    if (option == 'KNN'):
            model = KNeighborsRegressor(n_neighbors=15)
  
    return model

#------------------------------------------------------------------------------
def fit_Reg_model_feature_weights(model,X_Train,Y_Train, option,input_dim):
    Weights = []
    if option =='MLP' or option == 'CNN':
        # fit
        
        model = deep_fit(model, X_Train, Y_Train,input_dim)
        # weights of Deep
        #W = model.weights[0][:]
        #for i in range(np.shape(W)[0]):  # average weights of layer 1
         #   Weights.append(np.mean(W[i])) 
        
              
    if (option == 'DTree' or option == 'RF' or option == 'xgboost' or option =='Adaboost'):
         model =model.fit(X_Train, Y_Train)
         Weights = model.feature_importances_
         
    if (option == 'KNN'):
        model = model.fit(X_Train, Y_Train)
         
              
              

            
    return model, Weights
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def MLP_reg(model_info):
    # model create
    model = tf.keras.models.Sequential()
    model.add (tf.keras.layers.Dense(input_dim = model_info[0], units=model_info[2]
                                     , activation= 'relu'))
    for i in range(model_info[1]):
        model.add(tf.keras.layers.Dense(units = model_info[2]/(2**(i+1)), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
   
    model.add(tf.keras.layers.Dense(1,activation=None))
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=tf.keras.metrics.RootMeanSquaredError())# 'categorical_crossentropy'
    model.summary()
    return model
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def CNN_reg(model_info):
    model = tf.keras.models.Sequential()
    model.add (tf.keras.layers.Input(shape =(model_info[0],1), name='input'))
    for i in range(model_info[1]):
        model.add (tf.keras.layers.Conv1D(filters=model_info[2]/(2**(i)),
                                          kernel_size=3,
                                      padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
    
    
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  metrics=tf.keras.metrics.RootMeanSquaredError())# 'categorical_crossentropy'
    model.summary()
    return model  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def deep_fit(model,X_Train,Y_Train,input_dim):
    X_Train = np.reshape(X_Train, newshape=(len(Y_Train),input_dim,1))
    
    hist = model.fit(X_Train, Y_Train, epochs =30 , verbose = 1)# batch_size = 64)
    hist = hist.history
    fig = plt.figure(figsize= (12,6))
    plt.plot(hist['root_mean_squared_error'], lw=3)
    plt.ylabel('Epoch', size=15)
    plt.legend(['train RMSE'], fontsize=14)
    plt.show()
    
    return model
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def set_model_info(reg_model, input_dim):
    if reg_model =="MLP":
        # model_info=input_dim,num_layer, units
        model_info= [input_dim,4,128]
    if reg_model =="CNN":    
        # model_info=input_dim,num_layer, filters
        model_info= [input_dim,2,8]
    if  reg_model == "RF":
       # model_info=num_estimatore, depth
       model_info= [50,8]
    
    return model_info
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#!- Read datase
X, Y ,ID= Prepare_data()

'''#2 - aggrogate duplicates
X,Y = handle_duplicate(X,Y,ID)
np.savetxt('X.csv', X, delimiter=',')
np.savetxt('Y.csv', Y, delimiter=',')'''

'''# Read Data
X = np.loadtxt('X.csv', delimiter=',')
Y = np.loadtxt('Y.csv', delimiter=',')
print("new dataset shape:", np.shape(X))
for i in range(len(Y)):
    Y[i]= math.exp(-1*Y[i])
Y = np.round(Y,2)

# 3- Train and Test
num_Bins =3
Model_Option = 'Both'
Option_Base_reg = 'MLP'
model_info=set_model_info(Option_Base_reg, np.shape(X)[1])
num_Models= 6
Meta_Learner='RF'
model_info_AG=set_model_info(Meta_Learner, np.shape(X)[1])
CV(X,Y,num_Bins,Model_Option,Option_Base_reg,num_Models,Meta_Learner)'''
