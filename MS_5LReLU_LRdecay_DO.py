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



# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, imageSize1*imageSize1]   # imageSize1*imageSize1 = imageSize1*imageSize1
# \x/x\x/x\x/x\x/x\x/ ✞     -- fully connected layer (relu+dropout) W1 [imageSize1*imageSize1, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/ ✞       -- fully connected layer (relu+dropout) W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/ ✞         -- fully connected layer (relu+dropout) W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/ ✞           -- fully connected layer (relu+dropout) W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]
 
import math



try:
    import tensorflow as tf
except:
    import tf



tf.set_random_seed(0.0)

 
import numpy as np
import os
import matplotlib.pyplot as plt   
  
    
runfile('/Users/mulugetasemework/Dropbox/Phyton/processDataAndSetup.py', wdir='/Users/mulugetasemework/Dropbox/Phyton')   
   

# input X: imageSize1ximageSize1 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, imageSize1, imageSize1 ])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, n_classes])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# five layers and their number of neurons (tha last layer has n_classes softmax neurons)
L = 200
M = 100
N = 60
O = 30



# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([imageSize1*imageSize1,L], stddev=0.1))  # imageSize1*imageSize1 = imageSize1 * imageSize1
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)

W5 = tf.Variable(tf.truncated_normal([O, 1], stddev=0.1))
B5 = tf.Variable(tf.zeros([n_classes]))

# The model, with dropout at each layer
XX = tf.reshape(X, [-1, imageSize1*imageSize1])

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100


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

max_learning_rate = 0.01 
min_learning_rate = 0.0001 
decay_speed = 2#round(epochs/10)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_train_data, update_test_data, update_valid_data):
 
    thisCountTr = return_counterUpdateTr()
    start = thisCountTr[-1]
    end =     start +  batch_size
    batch_X,batch_Y = train_features[start:end], train_labels[start:end] 
 
    max_learning_rate = 0.01 
    min_learning_rate = 0.0001 
  
    decay_speed = 2#round(epochs/10)
 
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    
    # compute training values  
    if update_train_data:
        a, Train_cost,   w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": |--------- " + str(a) +   " --- " + str(Train_cost) +
              " --- <-Training accuracy:" +   " <- loss: "  + " : epoch " +
              str(i*100//len(train_features)+1)  + " (lr:" + str(learning_rate) + ")")
  
        return_train_cost(Train_cost)
        return_train_accuracy(a,i,testEvery)
        if TransormTrainingData==1:
            if end <= len(train_features_trans): 
                batch_X_trans,batch_Y_trans = train_features_trans[start:end], train_labels_trans[start:end]  
                a_trans, Train_cost_trans,   w_trans, b_trans = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X_trans, Y_: batch_Y_trans, pkeep: 1.0})
                return_train_cost_trans(Train_cost_trans)
                return_train_accuracy_trans(a_trans)

   
    if update_valid_data and doNotValidate == 0:
        startV = i
        end =   startV + 1
        batch_X_valid,batch_Y_valid = valid_features[startV:end], valid_labels[startV:end] 
      
        a, valid_cost,   w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X_valid, Y_: batch_Y_valid, pkeep: 1.0})
        print(str(i) + ":*** Validation accuracy:" + str(a) + " loss: " +
              str(valid_cost) + " (lr:" + str(learning_rate) + ")")
        return_valid_cost(valid_cost)
        return_valid_accuracy(a,i)   
    
    if update_test_data: 
        thisCount = return_counterUpdate()
        startTst = thisCount[-1]
        end =     startTst  + test_batch_size
        if end <=len( test_features):
            batch_X_test,test_labels2 = test_features[startTst:end], test_labels[startTst:end] 
     
            a, cost  = sess.run([accuracy, cross_entropy ], {X: batch_X_test, Y_: test_labels2, pkeep: 1.0})
            print(str(i) + ": ********* epoch " + str(i*100//len(test_features)+1) +
                  " ********* test accuracy:" + str(a) + " test loss: " + str(cost))
    
            return_test_cost(cost)
            return_test_accuracy(a,i)

            if test_thiscode==1:
                test_labels3 = np.array(swapped_test_labels[startTst:end])
                aS, ctestS  = sess.run([accuracy, cross_entropy ], {X: batch_X_test, Y_: test_labels3, pkeep: 1.0})
  
                return_test_costS(ctestS)
                return_test_accuracyS(aS,i)
    
        if test_shuffled == 1:
            thisCount =  return_counterUpdate_shuff_test()
            startTst_shuff = thisCount[-1]
            end_shuff =     startTst_shuff  + test_batch_size
            if  end_shuff <= len(test_features): 
                test_labels_reversed = test_labels.iloc[::-1]
                test_features_reversed = test_features[::-1]
                batch_X_shuff,batch_Y_shuff = test_features[startTst_shuff: end_shuff], test_labels_reversed[startTst: end_shuff] 
    
                aS_shuff, ctestS_shuff  = sess.run([accuracy, cross_entropy ], {X: (batch_X_shuff), Y_:  (batch_Y_shuff), pkeep: 1.0})
                
                return_test_cost_shuff(ctestS_shuff)
                return_test_accuracy_shuff(aS_shuff,i)     
            
        thisCount = return_counterUpdate_trans()
        startTst_trans = thisCount[-1]
        end_trans =     startTst_trans  + test_batch_size_trans
        if  end_trans <= len(test_features_trans): 
            batch_X_test_trans,test_labels2_trans = test_features_trans[startTst_trans:end_trans], test_labels_trans[startTst_trans: end_trans] 
  
     
            a_trans, cost_trans  = sess.run([accuracy, cross_entropy ], {X: batch_X_test_trans, Y_: test_labels2_trans, pkeep: 1.0})
   
            return_test_cost_trans(cost_trans)
            return_test_accuracy_trans(a_trans,i,testEvery_trans)     
  
     
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})

for i in range(epochs): training_step(i, i , i % testEvery == 0, i % validateEvery==0)


runfile('/Users/mulugetasemework/Dropbox/Phyton/plotDLs.py', wdir='/Users/mulugetasemework/Dropbox/Phyton')


mainTitle2='5L_ReLU_lrDecay_DO--' + 'TransformTrainingData:' + str(TransormTrainingData
) +'.svg'
mainTitle='five_layers_relu_lrdecay_dropout '+ '   ******* Translate: ' + str(translateImage
 )+ '    Rotate: ' + str(rotateImage)+ '   Affine: ' + str(affineOrNot
)+ '   Perspective: ' + str(perspectiveOrNot)+ '   Warp: ' + str(WarpOrNot
) + '   keepDataLength:   ' + str(keepDataSize
) + '   TransformTrainingData:   ' + str(TransormTrainingData
) + ' \n  max_learning_rate :   ' + str(max_learning_rate
)+ '   min_learning_rate:   ' + str(min_learning_rate) +  '   decay_speed:  ' + str(decay_speed)

figDir="/Users/mulugetasemework/Documents/Python/"
 
figname= mainTitle+'.svg'

f.suptitle(mainTitle,size=7 ) 
plt.subplots_adjust(left=0.1, wspace=0.2, top=0.7, bottom=0.2)
f.show()
os.chdir(figDir)

#plt.savefig(mainTitle2, format='svg', dpi=1200)    