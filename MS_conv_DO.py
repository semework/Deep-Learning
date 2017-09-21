#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:13:57 2017

@author: mulugetasemework
"""


# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

 
import math
import sys
import random
from sklearn.utils import shuffle

import cv2
import scipy.misc
#from tensorflow.examples.tutorials.mnist import input_data as mnist_data
#import cv2

try:
    import tensorflow as tf
except:
    import tf
#print("Tensorflow version " + tf.__version__)


tf.set_random_seed(0.0)


    
import pandas as pd
 
import numpy as np
import os
import matplotlib.pyplot as plt   


    
    
runfile('/Users/.../Phyton/processDataAndSetup.py', wdir='/Users/.../Phyton')   

# input X: image
# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>6 stride 1        W1 [5, 5, 1, 6]        B1 [6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]        B2 [12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]       B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None,imageSize1, imageSize1,1 ])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, n_classes])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([int(imageSize1/4) * int(imageSize1/4) * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(1, tf.float32, [N]))
#W5 = tf.Variable(tf.truncated_normal([N, n_classes], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=0.1))
B5 = tf.Variable(tf.constant(1, tf.float32, [n_classes]))

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, int(imageSize1/4) * int(imageSize1/4)* M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)


# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


    
max_learning_rate = 0.001 
min_learning_rate = 0.0001 
 
decay_speed = 1#round(epochs/10)


# You can call this function in a loop to train the model, 100 images at a time
#def training_step(i, update_test_data, update_train_data):
def training_step(i, update_train_data, update_test_data, update_valid_data):
 
    thisCountTr = return_counterUpdateTr()
    start = thisCountTr[-1]
    end =     start +  batch_size
    batch_X,batch_Y = train_features[start:end], train_labels[start:end]  
    batch_X = np.reshape( batch_X,[len(batch_X),imageSize1,imageSize1,-1])

    
    max_learning_rate = 0.1#0.02
    min_learning_rate = 0.00001#0.0000001
 
    decay_speed = 1#round(epochs/10)
 
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c,  w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X:batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": |--------- " + str(a) +   " --- " + str(c) +
              " --- <-Training accuracy:" +   " <- loss: "  + " : epoch " +
              str(i*100//len(train_features)+1)  + " (lr:" + str(learning_rate) + ")")
        return_train_cost(c)
        return_train_accuracy(a,i,testEvery) 
    
        if TransormTrainingData==1:
            if end <= len(train_features_trans): 
                batch_X_trans,batch_Y_trans = train_features_trans[start:end], train_labels_trans[start:end]  
                batch_X_trans = np.reshape( batch_X_trans,[len(batch_X_trans),imageSize1,imageSize1,-1])
                a_trans, c_trans,  w_trans, b_trans = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X:batch_X_trans, Y_: batch_Y_trans, pkeep: 1.0})
                return_train_cost_trans(c_trans)
                return_train_accuracy_trans(a_trans)  
    
    if update_valid_data and doNotValidate == 0:
        startV = i
        end =   startV + 1
        batch_X_valid,batch_Y_valid = valid_features[startV:end], valid_labels[startV:end] 
        batch_X_valid  = np.reshape(batch_X_valid,[batch_X_valid.shape[0],imageSize1,imageSize1,-1])
        a, valid_cost,   w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X_valid, Y_: batch_Y_valid, pkeep: 1.0})
        print(str(i) + ":*** Validation accuracy:" + str(a) + " loss: " + str(valid_cost) + " (lr:" + str(learning_rate) + ")")
        return_valid_cost(valid_cost)
        return_valid_accuracy(a,i)

    if update_test_data: 
        thisCount = return_counterUpdate()
        startTst = thisCount[-1]
        end =     startTst  + test_batch_size
        if end <= len(test_features):
            batch_X_test,test_labels2 = test_features[startTst:end], test_labels[startTst:end] 
            test_features2 = np.reshape(batch_X_test,[len(batch_X_test),imageSize1,imageSize1,-1])
     
            a, c  = sess.run([accuracy, cross_entropy ], {X: test_features2 , Y_: test_labels2, pkeep: 1.0})
            print(str(i) + ": ********* epoch " + str(i*100//test_features2.shape[0]+1) +
                  " ********* test accuracy:" + str(a) + " test loss: " + str(c))
    
            return_test_cost(c)
            return_test_accuracy(a,i)
        
            if test_thiscode==1:
                test_labels3 = swapped_test_labels[startTst:end] 
                aS, ctestS  =  sess.run([accuracy, cross_entropy ], {X: test_features2 , Y_: test_labels3, pkeep: 1.0})
                 
                  
                return_test_costS(ctestS)
                return_test_accuracyS(aS,i)    
        if test_shuffled == 1:
            thisCount =  return_counterUpdate_shuff_test()
            startTst_shuff = thisCount[-1]
            end_shuff =     startTst_shuff  + test_batch_size
            if  end_shuff <= len(test_features): 
                test_labels_reversed = test_labels.iloc[::-1]
                test_features_reversed = test_features[::-1]
                batch_X_shuff,batch_Y_shuff = test_features_reversed[startTst_shuff: end_shuff], test_labels[startTst: end_shuff] 
                batch_X_shuff = np.reshape(batch_X_shuff,[len(batch_X_shuff),imageSize1,imageSize1,-1])
    
                aS_shuff, ctestS_shuff  = sess.run([accuracy, cross_entropy ], {X: (batch_X_shuff), Y_:  (batch_Y_shuff), pkeep: 1.0})
                
                return_test_cost_shuff(ctestS_shuff)
                return_test_accuracy_shuff(aS_shuff,i)              
        
        thisCount = return_counterUpdate_trans()
        startTst_trans = thisCount[-1]
        end_trans =     startTst_trans  + test_batch_size_trans
        if  end_trans <= len(test_features_trans): 
            batch_X_test_trans,test_labels2_trans = test_features_trans[startTst_trans:end_trans], test_labels_trans[startTst_trans: end_trans] 
            test_features2_trans = np.reshape(batch_X_test_trans,[len(batch_X_test_trans),imageSize1,imageSize1,-1])
     
            a_trans, c_trans  = sess.run([accuracy, cross_entropy ], {X: test_features2_trans , Y_: test_labels2_trans, pkeep: 1.0})
            return_test_cost_trans(c_trans)
            return_test_accuracy_trans(a_trans,i,testEvery_trans)     

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

for i in range(epochs): training_step(i, i , i % testEvery == 0, i % validateEvery==0)

runfile('/Users/.../Phyton/plotDLs.py', wdir='/Users/.../Phyton')

mainTitle2='ConvBigDO--' + 'TransformTrainingData:' + str(TransormTrainingData
) +'.svg'

mainTitle='Convolutional_bigger_dropout'+ '   ******* Translate: ' + str(translateImage
 )+ '    Rotate: ' + str(rotateImage)+ '   Affine: ' + str(affineOrNot
)+ '   Perspective: ' + str(perspectiveOrNot)+ '   Warp: ' + str(WarpOrNot
) + '   keepDataLength:   ' + str(keepDataSize
) + '   TransformTrainingData:   ' + str(TransormTrainingData
) + ' \n  max_learning_rate :   ' + str(max_learning_rate
)+ '   min_learning_rate:   ' + str(min_learning_rate) +  '   decay_speed:  ' + str(decay_speed)


figDir="/Users/m.../Python/"
 
figname= mainTitle+'.svg'

f.suptitle(mainTitle,size=7 ) 
plt.subplots_adjust(left=0.1, wspace=0.2, top=0.7, bottom=0.2)
f.show()
os.chdir(figDir)



#plt.savefig(mainTitle2, format='svg', dpi=1200)    
