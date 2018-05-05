# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 00:00:09 2018

@author: moham
"""
import csv
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
from nltk.stem.porter import *
import keras
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense,\
                         Flatten,Conv2D, Embedding,SimpleRNN
from keras.layers import Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt

# Open File:
txt_file_test = r"2018-E-c-En-test.txt"
txt_file_train = r"2018-E-c-En-train.txt"
txt_file_dv = r"2018-E-c-En-dev.txt"
def preprocessing (txt_file_train, txt_file_dv,txt_file_test=None ,test_set=False):
    
    # convert to data frame
    df_train = pd.read_csv(txt_file_train, sep="\t")
    df_train['set']="train"
    df_dv = pd.read_csv(txt_file_dv, sep="\t")
    df_dv['set']="dv"
    df_concat =[df_train,df_dv]
    
    # check if there is any test set
    if not pd.isnull(txt_file_test):
       df_test = pd.read_csv(txt_file_test, sep="\t")   
       df_test['set']="test"
       df_concat.append(df_test)
       
    #concat every thing
    dfz = pd.concat(df_concat,ignore_index=True)   
    
    #find index of each set 
    index_train = dfz.query('set=="train"').index.values
    index_dv = dfz.query('set=="dv"').index.values
    index_test = dfz.query('set=="test"').index.values
    
        
    #tokenize
   
    tw_cm = TweetTokenizer(strip_handles= False)
    tokenized_tw =[ tw_cm.tokenize(cm) for cm in dfz['Tweet'] ]
    
    #make dictionary
    temmer = PorterStemmer()
    all_vocab = sorted(set([ temmer.stem(word) for post in tokenized_tw 
             for word in post]))
    #all_vocab = [word for word in ]
    
    size_of_vocab = len(all_vocab)
        
    token_to_index = {c: i for i, c in enumerate(all_vocab)}
    
    #make index from tweets
    indexd_tw=[]
    
    for tws in tokenized_tw:
        indexd_tw.append([token_to_index[temmer.stem(x)] for x in tws])
    #find the maximum size of comments    
    max_sizes=max([len(tws) for tws in tokenized_tw])
    
    matrix_input = np.zeros((len(tokenized_tw),max_sizes))
    # make padding for input
    for i in range(len(tokenized_tw)):
        for ind,ind_word in enumerate(indexd_tw[i]):
         matrix_input [i,ind] = ind_word  
         
    matrix_output = np.array(dfz.iloc[:,2:13]) 
    
    # slice files for different sets
    matrix_input_train = matrix_input[index_train,:]
    matrix_output_train = matrix_output[index_train,:] 
    matrix_input_dev = matrix_input[index_dv,:]
    matrix_output_dev = matrix_output[index_dv,:]
    matrix_input_test = matrix_input[index_test,:]
    matrix_output_test = matrix_output[index_test,:]
    
        
    return( matrix_input_train, matrix_output_train, matrix_input_dev,\
            matrix_output_dev, matrix_input_test, matrix_output_test,size_of_vocab)
    
def cnn_twiter(n_input, n_out, input_dim,units_activation = 'relu', batch_size =40 ):

    n_filters = 30
    embedin_size_out = min(50,input_dim/2 )
    model = Sequential()
    model.add(Embedding(input_dim = input_dim, input_length= n_input, output_dim= embedin_size_out ))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters= n_filters, kernel_size=4,activation= 'linear',strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(60,activation=units_activation))
       # model.add(Dense(60,activation=units_activation))
    model.add(Dropout(0.5))
    model.add(Dense(n_out,activation='sigmoid'))
    callsback = EarlyStopping(patience =2 )
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    #dict_1={'callbacks':[callsback]}
    dict_1={'callbacks':[callsback],'batch_size':batch_size}
    return(model, dict_1)


matrix_input_train, matrix_output_train, matrix_input_dev,\
            matrix_output_dev, matrix_input_test, matrix_output_test,size_of_vocab= \
            preprocessing (txt_file_train, txt_file_dv,txt_file_test) 
            
n_input =  matrix_input_train.shape[1] 
input_dim = size_of_vocab 
n_out =  matrix_output_train.shape[1]

model_cnn, kwargs=cnn_twiter(n_input, n_out, input_dim,units_activation = 'tanh'\
                          , batch_size =1  )
kwargs.update(x=matrix_input_train,y=matrix_output_train,epochs=20, \
              validation_data=(matrix_input_dev, matrix_output_dev))

hist_cnn = model_cnn.fit(**kwargs)
u_cnn=model_cnn.predict(matrix_input_dev)
j_cnn=jaccard_similarity_score(np.round(u_cnn), matrix_output_dev.astype(int))

#model_048test = model_cnn.to_json()
#with open("model_048_test.json","w") as f:
#    f.write(model_048test)
#model_cnn.save_weights("model_048test.h5")

u_cnn_test=np.round(model_cnn.predict(matrix_input_test))
u_cnn_test = u_cnn_test.astype(int)

# concatenate everything
df_test = pd.read_csv(txt_file_test,sep="\t")
df_test.iloc[:,2:]= u_cnn_test
