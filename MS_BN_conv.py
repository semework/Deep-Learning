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

# import sys
#
#sys.path.append('/Users/mulugetasemework/Dropbox/Phyton')
#
#import TransformInputsDef.py
#from subprocess import call
#call(["python", "/Users/mulugetasemework/Dropbox/Phyton/TransformInputsDef.py"]) 


import math
 
#from tensorflow.examples.tutorials.mnist import input_data as mnist_data
#import cv2

try:
    import tensorflow as tf
except:
    import tf
#print("Tensorflow version " + tf.__version__)


tf.set_random_seed(0.0)

 
import numpy as np
import os
import matplotlib.pyplot as plt   


runfile('/Users/mulugetasemework/Dropbox/Phyton/processDataAndSetup.py', wdir='/Users/mulugetasemework/Dropbox/Phyton')    


# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                    X [batch, imageSize1, imageSize1, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 6x6x1=>24 stride 1      W1 [5, 5, 1, 24]        B1 [24]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, imageSize1, imageSize1, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x6=>48 stride 2      W2 [5, 5, 6, 48]        B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer +BN 4x4x12=>64 stride 2     W3 [4, 4, 12, 64]       B3 [64]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                    Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
#        · · ·                                                     Y [batch, 10]

# input X: imageSize1ximageSize1 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, imageSize1, imageSize1,1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, n_classes])
# variable learning rate
lr = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([int(imageSize1/4) * int(imageSize1/4) * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 1], stddev=0.1))
#W5 = tf.Variable(tf.truncated_normal([N, n_classes], stddev=0.1))
B5 = tf.Variable(tf.constant(1, tf.float32, [n_classes]))

# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is imageSize1ximageSize1
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
stride = 2  # output is 14x14
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
stride = 2  # output is 7x7
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, int(imageSize1/4) * int(imageSize1/4) * M])

Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
conv_activations = tf.concat([tf.reshape(tf.reduce_max(Y1r, [0]), [-1]), tf.reshape(tf.reduce_max(Y2r, [0]), [-1]), tf.reshape(tf.reduce_max(Y3r, [0]), [-1])], 0)
dense_activations = tf.reduce_max(Y4r, [0])
 
# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# learning rate decay
max_learning_rate = 0.2 
min_learning_rate = 0.00001 
 
decay_speed =  2#round(epochs/5)

# You can call this function in a loop to train the model, 100 images at a time
#def training_step(i, update_test_data, update_train_data):
def training_step(i, update_train_data, update_test_data, update_valid_data):
 
    thisCountTr = return_counterUpdateTr()
    start = thisCountTr[-1]
    end =     start +  batch_size
 
    batch_X,batch_Y = train_features[start:end], train_labels[start:end]  
    batch_X = np.reshape( batch_X,[len(batch_X),imageSize1,imageSize1,-1])

    # learning rate decay
 
    max_learning_rate = 0.2 
    min_learning_rate = 0.00001 
 
    decay_speed =  2#round(epochs/5)
 
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
  
    # compute training values  
    if update_train_data:
        a, ctrain,  ca, da = sess.run([accuracy, cross_entropy,  conv_activations, dense_activations], {X: batch_X, Y_: batch_Y, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
        print(str(i) + ": |--------- " + str(a) +   " --- " + str(ctrain) +
              " --- <-Training accuracy:" +   " <- loss: "  + " : epoch " + str(i) + " (lr:" + str(learning_rate) + ")")
        return_train_cost(ctrain)
        return_train_accuracy(a,i,testEvery)
    
        if TransormTrainingData == 1:
            thisCountTr = return_counterUpdate_trans()
            start = thisCountTr[-1]
            end =     start +  batch_size
            if end <= len(train_features_trans): 
                batch_X_trans,batch_Y_trans = train_features_trans[start:end], train_labels_trans[start:end]  
                batch_X_trans = np.reshape( batch_X_trans,[len(batch_X_trans),imageSize1,imageSize1,-1])

                a_trans, ctrain_trans,  ca_trans, da_trans = sess.run([accuracy, cross_entropy,  conv_activations, dense_activations], {X: batch_X_trans, Y_: batch_Y_trans, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
                print(str(i) + ": |----trans----- " + str(a_trans) +   " --- " + str(ctrain_trans) +
                      " --- <-Training accuracy:" +   " <- loss: "  + " : epoch " + str(i) + " (lr:" + str(learning_rate) + ")")
                return_train_cost_trans(ctrain_trans)
                return_train_accuracy_trans(a_trans)
 
        if  test_shuffled == 1:           
            thisCountTr = return_counterUpdate_shuff()
            startTrS = thisCountTr[-1]
            endTrS =     startTrS +  batch_size
            if endTrS <= len(train_features_shuff): 
                batch_X_shuff,batch_Y_shuff = train_features[startTrS:endTrS], train_labels_shuff[startTrS:endTrS]  
                batch_X_shuff = np.reshape( batch_X_shuff,[len(batch_X_shuff),imageSize1,imageSize1,-1])
                a_shuff, ctrain_shuff,  ca_shuff, da_shuff = sess.run([accuracy, cross_entropy,  conv_activations, dense_activations], {X: batch_X_shuff, Y_: batch_Y_shuff, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
                print(str(i) + ": |----trans----- " + str(a_shuff) +   " --- " + str(ctrain_shuff) +
                      " --- <-Training accuracy:" +   " <- loss: "  + " : epoch " + str(i) +
                      " (lr:" + str(learning_rate) + ")")
                return_train_cost_shuff(ctrain_shuff)
                return_train_accuracy_shuff(a_shuff,i,testEvery)

    if update_valid_data and doNotValidate == 0:
        startV = i
        end =   startV + 1
        batch_X_valid,batch_Y_valid = valid_features[startV:end], valid_labels[startV:end] 
        batch_X_valid  = np.reshape(batch_X_valid,[batch_X_valid.shape[0],imageSize1,imageSize1,-1])
        a, valid_cost,  ca, da = sess.run([accuracy, cross_entropy,  conv_activations, dense_activations], {X: batch_X_valid, Y_: batch_Y_valid, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
        print(str(i) + ":Validation accuracy:" + str(a) + " loss: " + str(valid_cost) +
              " (lr:" + str(learning_rate) + ")") 
        return_valid_cost(valid_cost)
        return_valid_accuracy(a,i)
    if update_test_data: 
        thisCount = return_counterUpdate()
        startTst = thisCount[-1]
        end =     startTst  + test_batch_size
        if end <= len(test_features): 
            batch_X_test,test_labels2 = test_features[startTst:end], test_labels[startTst:end] 
            test_features2 = np.reshape(batch_X_test,[len(batch_X_test),imageSize1,imageSize1,-1])
            
            
            a, ctest  = sess.run([accuracy, cross_entropy ], {X: test_features2, Y_: test_labels2, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
            thisLoss = 100
            thisLoss = [ctest/thisLoss if i!=0   else thisLoss] 
            print(str(i) + ": |---- " + str(a) +   " --- " + str(ctest) +
                  " ---------- <-Testing accuracy:" +   " <- loss: "  + " : epoch " + str(i) )
       
            return_test_cost(ctest)
            return_test_accuracy(a,i)
      #trans
            if test_thiscode==1:
                test_labels3 = swapped_test_labels[startTst:end] 
                aS, ctestS  = sess.run([accuracy, cross_entropy ], {X: test_features2, Y_: test_labels3, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
                
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
                batch_X_shuff = np.reshape(batch_X_shuff,[len(batch_X_shuff),imageSize1,imageSize1,-1])
    
                aS_shuff, ctestS_shuff  = sess.run([accuracy, cross_entropy ], {X:  (batch_X_shuff), Y_:  (batch_Y_shuff), tst: True, pkeep: 1.0, pkeep_conv: 1.0})
                
                return_test_cost_shuff(ctestS_shuff)
                return_test_accuracy_shuff(aS_shuff,i)

        thisCount = return_counterUpdate_trans()
        startTst_trans = thisCount[-1]
        end_trans =     startTst_trans  + test_batch_size_trans
        if  end_trans <= len(test_features_trans): 
            batch_X_test_trans,test_labels2_trans = test_features_trans[startTst_trans:end_trans], test_labels_trans[startTst_trans: end_trans] 
            test_features2_trans = np.reshape(batch_X_test_trans,[len(batch_X_test_trans),imageSize1,imageSize1,-1])
        

            a_trans, ctest_trans  = sess.run([accuracy, cross_entropy ], {X: test_features2_trans, Y_: test_labels2_trans, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
            thisLoss_trans = 100
            thisLoss_trans = [ctest_trans/thisLoss_trans if i!=0   else thisLoss_trans] 
            print(str(i) + ": |---- " + str(a_trans) +   " --- " + str(ctest_trans) +
                  " ---------- <-Testing accuracy:" +   " <- loss: "  + " : epoch " + str(i) )
       
            return_test_cost_trans(ctest_trans)
            return_test_accuracy_trans(a_trans,i,testEvery_trans)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})

for i in range(epochs): training_step(i, i , i % testEvery == 0, i % validateEvery==0)
 
runfile('/Users/mulugetasemework/Dropbox/Phyton/plotDLs.py', wdir='/Users/mulugetasemework/Dropbox/Phyton')

mainTitle2='BN_conv--' + 'TransformTrainingData:' + str(TransormTrainingData
) +'.svg'

mainTitle='4.2_batchnorm_convolutional'+ '   ******* Translate: ' + str(translateImage
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

plt.savefig(mainTitle2, format='svg', dpi=1200)   

 