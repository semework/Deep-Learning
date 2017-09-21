#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:12:54 2017

@author: mulugetasemework. 

This code imports raw neural data (characteristics
from pickle forma which was split and saved by the code: "neuroPickle.py".
It makes use of another code:"TransformInputsDef.py" which does geometric 
transfomations to "corrupt" the data and increase the size of training and test
sets)

It is also a master code called by neural net codes as it sets up epoch sizes, 
has the functions to perfrom and keep track of training and test accuracy and 
cost append(ing)
"""


from sklearn.utils import shuffle

tf.set_random_seed(0.0)

try:
    import cPickle as pickle
except:
    import pickle
    
import pandas as pd
 
import numpy as np
import os


reshapeImages = 1    
 
tf.set_random_seed(0)
seed = 128
rng = np.random.RandomState(seed)

path = "/Users/mulugetasemework/Documents/Python"
os.chdir(path)

plt.close("all")
saved_neuro_pickle = pickle.load(open( "neuro_pickle_file.pickle", "rb" ))

train_features,train_labels,valid_features,valid_labels,test_features,test_labels = saved_neuro_pickle['train_features'],saved_neuro_pickle['train_labels'],saved_neuro_pickle['valid_features'],saved_neuro_pickle['valid_labels'],saved_neuro_pickle['test_features'],saved_neuro_pickle['test_labels']
#print('Training set', train_features.shape, train_labels.shape)
#print('Validation set', valid_features.shape, valid_labels.shape)
#print('Test set', test_features.shape, test_labels.shape)

doNotValidate = 1

def ConsolidateTrainAndValid(train_features,train_labels,valid_features,valid_labels):
    train_features = np.vstack([train_features,valid_features])
    train_labels= np.vstack([train_labels,valid_labels])
    return train_features,train_labels 

if doNotValidate==1:
    train_features,train_labels  = ConsolidateTrainAndValid(train_features,train_labels,valid_features,valid_labels)

print("Main training label size :" + str(train_labels.shape))

if reshapeImages==1:
    reshapeSize = 32
    train_features = np.resize(train_features,[train_features.shape[0],reshapeSize,reshapeSize])
    test_features = np.resize(test_features,[test_features.shape[0],reshapeSize,reshapeSize])
    valid_features = np.resize(valid_features,[valid_features.shape[0],reshapeSize,reshapeSize])
    
   
imageSize1 = train_features.shape[2]
imageSize1= int(imageSize1)
imageSize2=(imageSize1/2)

n_input= imageSize1*imageSize1  
n_classes= 2  #no/memory

rotationAngle = 5
RepeatTrainingData = 0
RepeatTestingData = 0


translateImage = 1
rotateImage = 1
affineOrNot = 1
perspectiveOrNot = 1
WarpOrNot = 1


test_thiscode = 1
test_shuffled = 0
trX,trY = 1, 2
TransormTrainingData = 0
keepDataSize = 1

train_features =  np.reshape(train_features, [-1, imageSize1,imageSize1])
valid_features =  np.reshape(valid_features, [-1, imageSize1,imageSize1])
test_features =  np.reshape(test_features, [-1, imageSize1,imageSize1])

def RepeatTestData(test_features,test_labels):
    test_features = np.vstack([test_features,test_features])
    test_labels= np.vstack([test_labels,test_labels])
    return test_features,test_labels 

if RepeatTestingData==1:
    test_features, test_labels = RepeatTestData(test_features,test_labels)


if TransormTrainingData==1: 
    train_features_trans, train_labels_trans = TransformInputsDef(train_features
    ,train_labels,imageSize1,translateImage,rotateImage,rotationAngle
    ,affineOrNot,perspectiveOrNot,WarpOrNot,keepDataSize)

test_features_trans, test_labels_trans = TransformInputsDef(test_features,
test_labels,imageSize1,translateImage,rotateImage,rotationAngle,
affineOrNot,perspectiveOrNot,WarpOrNot,keepDataSize)

if TransormTrainingData==1: 
    train_features, train_labels = train_features_trans, train_labels_trans
 

batch_size = 10

if TransormTrainingData==1 and keepDataSize == 0:
    batch_size = batch_size*5
 
    
epochs = round((len(train_features)/batch_size)) 
repData =  epochs*10

def ChooseEpochSize(epochs,RepeatTrainingData,train_features, 
                    train_labels,batch_size):
    if RepeatTrainingData == 0:
        epochs = round((len(train_features)/batch_size)-2)
        return epochs,train_features,train_labels
    else:
        train_features =  np.vstack([train_features]*repData)
        train_labels =  np.vstack([train_labels.astype(np.float32)]*repData)
        epochs = round((len(train_features)/batch_size)-2)
        return epochs,train_features,train_labels

epochs,train_features, train_labels= ChooseEpochSize(epochs,
RepeatTrainingData,train_features, train_labels,batch_size)
 

train_features,train_labels = shuffle(train_features,train_labels)
test_features,test_labels = shuffle(test_features,test_labels)

test_features_shuff,test_labels_shuff = shuffle(test_features,test_labels)
train_features_shuff,train_labels_shuff = shuffle(train_features,train_labels)

if TransormTrainingData==1: 
    train_features_trans,train_labels_trans = shuffle(train_features_trans,
                                                      train_labels_trans)
    
test_features_trans,test_labels_trans = shuffle(test_features_trans,
                                                test_labels_trans)

swapped_test_labels = pd.DataFrame()
if test_thiscode==1:
    if isinstance(test_labels, pd.DataFrame):
        swapped_test_labels = test_labels.values[:,[1, 0]]
    else: 
        swapped_test_labels = test_labels[:,[1, 0]]

 
 
test_batch_size = 20 

testEvery = round(epochs/(len(test_features)/test_batch_size))+1

test_batch_size_trans = 20
testEvery_trans = round(epochs/(len(test_features_trans)/test_batch_size_trans))+1
   
if test_batch_size <= 0:
    test_batch_size = 15
    if TransormTrainingData==1 and keepDataSize==0:
        test_batch_size = 20
    testEvery = round(epochs/(len(test_features)/test_batch_size))

 
if test_batch_size_trans <= 0:
    test_batch_size_trans = 15
    if TransormTrainingData==1 and keepDataSize==0:
          test_batch_size_trans = 20
    testEvery_trans = round(epochs/(len(test_features_trans)/test_batch_size_trans))
    
    
if TransormTrainingData==1 and keepDataSize == 0:
    test_batch_size = 10
    test_batch_size_trans = 20
    testEvery = round(epochs/round(len(test_features)/test_batch_size))+5
    testEvery_trans = round(epochs/(len(test_features_trans)/test_batch_size_trans))+3


validateEvery = 2# 

print("Test every: " + str(testEvery) + "     Validate every: " + 
      str(validateEvery)+ " Test batch size: " +str(test_batch_size) +
      " trans-test batch size: " +str(test_batch_size_trans)) 

###############################################################################
############################################################################### 
trCost=list()

def return_train_cost(cost):
    trCost.append(cost)
    return trCost 

trCost_trans=list()

def return_train_cost_trans(cost):
    trCost_trans.append(cost)
    return trCost_trans 

trAcc_trans=list()

def return_train_accuracy_trans(a):
    trAcc_trans.append(a)
    return trAcc_trans

validCost=list()

def return_valid_cost(cost):
    validCost.append(cost)
    return validCost

testCost=list()

def return_test_cost(cost):
    testCost.append(cost)
    return testCost 

testCost_trans=list()

def return_test_cost_trans(cost):
    testCost_trans.append(cost)
    return testCost_trans 

testAcc_trans=list()
testIndx_trans=list()   
startTst_trans=list()
def return_test_accuracy_trans(a,i,testEvery_trans):
    testAcc_trans.append(a)
    testIndx_trans.append(i)
    startTst_trans.append(i + testEvery_trans)
    return testAcc_trans,testIndx_trans,startTst_trans

trAcc=list()
trIndx=list()
trIndx.append(0)
startTst=list()
def return_train_accuracy(a,i,testEvery):
    trAcc.append(a)
    trIndx.append(i)
    startTst.append(i + testEvery)
    return trAcc,trIndx ,startTst

def return_testCount(startTst):       
    startV =startTst[len(startTst)-1]
    return startV

validAcc=list()
validIndx=list()

def return_valid_accuracy(a,i):
    validAcc.append(a)
    validIndx.append(i)
    return validAcc,validIndx

testAcc=list()
testIndx=list()

def return_test_accuracy(a,i):
    testAcc.append(a)
    testIndx.append(i)
    return testAcc,testIndx 

startV2=list()
def return_nowUpdate(validateEvery,i):
    startV2.append(i + validateEvery)
    return startV2

testValidCounter=list()
testValidCounter.append(0)
def return_counterUpdate():
    testValidCounter.append(testValidCounter[-1]+test_batch_size )
    return testValidCounter

testValidCounter_trans=list()
testValidCounter_trans.append(0)
def return_counterUpdate_trans():
    testValidCounter_trans.append(testValidCounter_trans[-1]+test_batch_size_trans)
    return testValidCounter_trans


testValidCounterTr=list()
testValidCounterTr.append(0)
def return_counterUpdateTr():
    testValidCounterTr.append(testValidCounterTr[-1] + batch_size)
    return testValidCounterTr

####### swapped test labels
testCostS=list()

def return_test_costS(cost):
    testCostS.append(cost)
    return testCostS 

testAccS=list()
testIndxS=list()
def return_test_accuracyS(a,i):
    testAccS.append(a)
    testIndxS.append(i)
    return testAccS,testIndxS 


trCost_shuff=list()

def return_train_cost_shuff(cost):
    trCost_shuff.append(cost)
    return trCost_shuff 

trAcc_shuff=list()
trIndx_shuff=list()
trIndx_shuff.append(0)
startTst_shuff=list()
def return_train_accuracy_shuff(a,i,testEvery):
    trAcc_shuff.append(a)
    trIndx_shuff.append(i)
    startTst_shuff.append(i + testEvery)
    return trAcc_shuff,trIndx_shuff,startTst_shuff

testCost_shuff=list()

def return_test_cost_shuff(cost):
    testCost_shuff.append(cost)
    return testCost_shuff 

testAcc_shuff=list()
testIndx_shuff=list()   
 
def return_test_accuracy_shuff(a,i):
    testAcc_shuff.append(a)
    testIndx_shuff.append(i)
    return testAcc_shuff,testIndx_shuff


testValidCounter_shuff=list()
testValidCounter_shuff.append(0)
def return_counterUpdate_shuff():
    testValidCounter_shuff.append(testValidCounter_shuff[-1]+test_batch_size )
    return testValidCounter_shuff

testValidCounter_shuff_test=list()
testValidCounter_shuff_test.append(0)
def return_counterUpdate_shuff_test():
    testValidCounter_shuff_test.append(testValidCounter_shuff_test[-1]+test_batch_size )
    return testValidCounter_shuff_test

if doNotValidate == 1:
    print("Not validating")
###############################################################################
############################################################################### 
