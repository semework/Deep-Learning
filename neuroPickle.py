#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:15:46 2017

@author: mulugetasemework

This code imports .csv neural files which have different neural characteristis
(colums) such as firing rate mean, median, max, response latency, etc. and 
splits the data into training and test sets and saves them as pickled files. 
"""


import numpy as np
import matplotlib.pyplot as plt
 
import os

import glob
import pandas as pd
try:
    import cPickle as pickle
except:
    import pickle
 
plt.close("all")

path = "/Users/mulugetasemework/Documents/Python"
os.chdir(path)
#bring in and catenate spreadsheets
all_files = glob.glob(os.path.join(path, "*.csv")) 


 
patDatah = "/Users/.../R2"
os.chdir(patDatah)
#df2 = pd.read_csv("1MetaTable1.csv")
#df2 = pd.read_csv("veryImpData.csv")
#df2 = pd.read_csv("veryVitalData.csv")
df2 = pd.read_csv("IncPurityTop10.csv")
#df2 = pd.read_csv("veryImpDataIncPurity.csv")

print("CSVs catenated")
print("incoming data size:  " + str (df2.shape))
os.chdir(path)

pMemoBaseMAX = df2['pMemoBaseMAX']
train_labels = (pMemoBaseMAX<0.05).astype(np.int64)

goodMems = (train_labels==1)
badMems = (train_labels==0)

df5 = df2.loc[df2['pMemoBaseMAX']<0.05]
 

badVsgoodRatio = round(sum(badMems.astype(np.int64))/sum(goodMems.astype(np.int64)))-1

repeatGoodMems = 0

if repeatGoodMems==1:
    goodMemsRepeated =   pd.concat([df5]*badVsgoodRatio.astype(np.int64))
    df2 =df2.append(goodMemsRepeated)
    
pMemoBaseMAX = df2['pMemoBaseMAX']

 
df2 = df2.iloc[:,:df2.shape[1]]


df2.drop(['pMemoBaseMAX'], axis = 1, inplace = True, errors = 'ignore')
 
Header = list(df2.columns.values)
leng = len(Header)
neurons = df2.shape[0]
df2 = df2.fillna(0)


plotFigs = 0
newMax = 255;
doNotScale = 0
df = pd.DataFrame(index=range(neurons), columns=Header)

for i in range(0,leng):
    a = abs(np.float32(np.array(df2.iloc[:,[i] ])))
    a = np.nan_to_num(a)
    if doNotScale == 0:
        if (a == 0).all() or (a == 1).all() :
             d = a
    #         print(str(np.amax(d)) + "--- " + str(i) + " Header:" + str(Header[i]))
        else:
             d = np.float32(( (a)/float(np.amax((a))))*newMax)
    #         print(str(np.amax(d)) + "--- " + str(i) + " Header:" + str(Header[i]))

        df[Header[i]]= d
    else:
        df[Header[i]]= a


patchLen= (round(np.sqrt(leng))*round(np.sqrt(leng)))-leng


width, height = round(np.sqrt(leng)), round(np.sqrt(leng))

if width*2 % 4 != 0:
    width,height = width+1,width+1


width = width.astype(np.int64)
height = height.astype(np.int64)
    
plt.ion()  
fig=plt.figure()

 
i=0

x=list()
y=list()
newMax = 255;
train_labels=[]
temp =  []
thisLabel = []


import matplotlib.pyplot as plt
plt.ion()
import numpy as np
plt.close("all")
fig, ax = plt.subplots()
def plotAndSaveImages(df,neurons,width, height,plotFigs):
    for i in range(neurons):
        a = abs(np.float32(np.array(df.iloc[i]))) 
 
        a = np.resize(a,(width, height))
 
        temp.append(np.float32(a))
 

        if plotFigs == 1:
 
            plt.imshow(a, interpolation='nearest')
            plt.pause(.0001)
         
            plt.show()
   
        print('Neuron ' + str(i) )
plotAndSaveImages(df,neurons,width, height,plotFigs)

 
dummyf = pd.DataFrame(np.random.randn(len(temp), 2), columns=list('AB'))

train_xs = np.stack(temp)

train_x1, val_x1, val_x1 = np.split(dummyf.sample(frac=0.66),
                                    [int(.33*len(train_xs)), int(.33* len(train_xs))])
 

train_test_Indices = list(np.concatenate((train_x1.index, val_x1.index), axis=0))
allLists = list(range(1, len(temp)))

testIndices =  list(set(allLists) - set(train_test_Indices))

train_x2, val_x2,test_x2  = train_xs[train_x1.index.values,:,:],train_xs[
        val_x1.index.values,:,:],  train_xs[testIndices,:,:]
 
train_labels = (pMemoBaseMAX<0.05).astype(np.int64)

train_labels2 = pd.DataFrame(np.stack(train_labels))

train_y2, val_y2, test_y2 = train_labels2.iloc[train_x1.index.values
    ],train_labels2.iloc[val_x1.index.values],  train_labels2.iloc[testIndices]

  

train_labels2 = (~train_y2.values.astype(bool)).astype(np.ndarray).astype(int)
val_labels2 = (~val_y2.values.astype(bool)).astype(np.ndarray).astype(int)
test_labels2 =  (~test_y2.values.astype(bool)).astype(np.ndarray).astype(int)

trLab=horizontalStack = pd.concat([pd.DataFrame(train_y2.values
                                    ),pd.DataFrame(train_labels2)], axis=1)
valLab=horizontalStack = pd.concat([pd.DataFrame(val_y2.values
                                 ),pd.DataFrame(val_labels2)], axis=1)
testLab=horizontalStack = pd.concat([pd.DataFrame(test_y2.values
                                  ),pd.DataFrame(test_labels2)], axis=1)


neuro_pickle_file = 'neuro_pickle_file.pickle'

try:
  f = open(neuro_pickle_file, 'wb')
  save = {
    'train_features': train_x2,
    'train_labels':  trLab,
    'valid_features': val_x2,
    'valid_labels':  valLab,
    'test_features': test_x2,
    'test_labels':  testLab,
    'train_labels22': train_labels2,
    'valid_labels22':  val_labels2,
    'test_labels22':  test_labels2,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', neuro_pickle_file, ':', e)
  raise

plt.close("all")
 

statinfo = os.stat(neuro_pickle_file)
print('Compressed pickle size:', statinfo.st_size)

