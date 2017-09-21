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
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (relu+BN)      W1 [imageSize1*imageSize1, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (relu+BN)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (relu+BN)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (relu+BN)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

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
 
 

try:
    import tensorflow as tf
except:
    import tf
#print("Tensorflow version " + tf.__version__)


tf.set_random_seed(0.0)

 
 
import numpy as np
import os
import matplotlib.pyplot as plt   

 
    
    
runfile('/Users/.../Phyton/processDataAndSetup.py', wdir='/Users/mulugetasemework/Dropbox/Phyton')   
    

# input X: imageSize1ximageSize1 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, imageSize1, imageSize1, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, n_classes])
# variable learning rate
lr = tf.placeholder(tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = n_classes

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/n_classes
W1 = tf.Variable(tf.truncated_normal([imageSize1*imageSize1, L], stddev=0.1))  # imageSize1*imageSize1 = imageSize1 * imageSize1
B1 = tf.Variable(tf.ones([L])/100)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/100)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/100)
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
B4 = tf.Variable(tf.ones([P])/100)
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.ones([Q])/100)

## Batch normalisation conclusions:
# On RELUs, you have to display batch-max(activation) to see the nice effect on distribution but
# it is very visible.
# With RELUs, the scale and offset variables can be omitted. They do not seem to do anything.

# Steady 98.5% accuracy using these parameters:
# moving average decay: 0.998 (equivalent to averaging over two epochs)
# learning rate decay from 0.03 to 0.0001 speed 1000 => max 98.59 at 6500 iterations, 98.54 at 10K it,  98% at 1300it, 98.5% at 3200it

# relu, no batch-norm, lr(0.003, 0.0001, 2000) => 98.2%
# relu, batch-norm lr(0.03, 0.0001, 1000) => 98.5% - 98.55%
# relu, batch-norm, no offsets => 98.5% - 98.55% (no change)
# relu, batch-norm, no scales => 98.5% - 98.55% (no change)
# relu, batch-norm, no scales, no offsets => 98.5% - 98.55% (no change) - even peak at 98.59% :-)

# Correct usage of batch norm scale and offset parameters:
# According to BN paper, offsets should be kept and biases removed.
# In practice, it seems to work well with BN without offsets and traditional biases.
# "When the next layer is linear (also e.g. `nn.relu`), scaling can be
# disabled since the scaling can be done by the next layer."
# So apparently no need of scaling before a RELU.
# => Using neither scales not offsets with RELUs.

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

# The model
XX = tf.reshape(X, [-1, imageSize1*imageSize1])

# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1)
Y1 = tf.nn.relu(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2)
Y2 = tf.nn.relu(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3)
Y3 = tf.nn.relu(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4 = tf.nn.relu(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1])], 0)
# to use for sigmoid
#allactivations = tf.concat([tf.reshape(Y1, [-1]), tf.reshape(Y2, [-1]), tf.reshape(Y3, [-1]), tf.reshape(Y4, [-1])], 0)
# to use for RELU
allactivations = tf.concat([tf.reduce_max(Y1, [0]), tf.reduce_max(Y2, [0]), tf.reduce_max(Y3, [0]), tf.reduce_max(Y4, [0])], 0)
alllogits = tf.concat([tf.reshape(Y1l, [-1]), tf.reshape(Y2l, [-1]), tf.reshape(Y3l, [-1]), tf.reshape(Y4l, [-1])], 0)
 

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

max_learning_rate = 0.2 
min_learning_rate = 0.00001 
 
decay_speed =  2#round(epochs/10)

def training_step(i, update_train_data, update_test_data, update_valid_data):
 
    thisCountTr = return_counterUpdateTr()
    start = thisCountTr[-1]
    end =     start +  batch_size
    batch_X,batch_Y = train_features[start:end], train_labels[start:end]  
    batch_X = np.reshape( batch_X,[len(batch_X),imageSize1,imageSize1,-1])

    max_learning_rate = 0.2 
    min_learning_rate = 0.00001 
 
    decay_speed =  2#round(epochs/10)
 
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values  
    if update_train_data:
        a, c,   al, ac = sess.run([accuracy, cross_entropy,  alllogits, allactivations], {X: batch_X, Y_: batch_Y, tst: False})
        print(str(i) + ": |--------- " + str(a) +   " --- " + str(c) +  " --- <-Training accuracy:" +
              " <- loss: "  + " : epoch " + str(i*100//len(train_features)+1)+ " (lr:" + str(learning_rate) + ")" )
  
        return_train_cost(c)
        return_train_accuracy(a,i,testEvery)
        if TransormTrainingData==1:
            if end <= len(train_features_trans): 
                batch_X_trans,batch_Y_trans = train_features_trans[start:end], train_labels_trans[start:end]  
                batch_X_trans = np.reshape( batch_X_trans,[len(batch_X_trans),imageSize1,imageSize1,-1])
                a_trans, c_trans,   al_trans, ac_trans = sess.run([accuracy, cross_entropy,  alllogits, allactivations], {X: batch_X_trans, Y_: batch_Y_trans, tst: False})
        
                return_train_cost_trans(c_trans)
                return_train_accuracy_trans(a_trans)
   

    if update_valid_data and doNotValidate == 0:
        startV = i
        end =   startV + 1
        batch_X_valid,batch_Y_valid = valid_features[startV:end], valid_labels[startV:end] 
        batch_X_valid  = np.reshape(batch_X_valid,[len(batch_X_valid),imageSize1,imageSize1,-1])
        
        a, valid_cost,  al, ac = sess.run([accuracy, cross_entropy,  alllogits, allactivations], {X: batch_X_valid, Y_: batch_Y_valid, tst: False})
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
            a, c  = sess.run([accuracy, cross_entropy ], {X: test_features2, Y_:  test_labels2, tst: True})
            print(str(i) + ": |---- " + str(a) +   " --- " + str(c) +  " ---------- <-Testing accuracy:" +
                  " <- loss: "  + " : epoch " + str(i*100//len(train_features)+1) )
       
            return_test_cost(c)
            return_test_accuracy(a,i)
            if test_thiscode==1:
                test_labels3 = swapped_test_labels[startTst:end] 
                aS, ctestS  =  sess.run([accuracy, cross_entropy ], {X: test_features2, Y_:  test_labels3, tst: True})
          
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
    
                aS_shuff, ctestS_shuff  = sess.run([accuracy, cross_entropy ], {X: (batch_X_shuff), Y_:  (batch_Y_shuff), tst: True})
                
                return_test_cost_shuff(ctestS_shuff)
                return_test_accuracy_shuff(aS_shuff,i)            
            
        thisCount = return_counterUpdate_trans()
        startTst_trans = thisCount[-1]
        end_trans =     startTst_trans  + test_batch_size_trans
        if  end_trans <= len(test_features_trans): 
            batch_X_test_trans,test_labels2_trans = test_features_trans[startTst_trans:end_trans], test_labels_trans[startTst_trans: end_trans] 
            test_features2_trans = np.reshape(batch_X_test_trans,[len(batch_X_test_trans),imageSize1,imageSize1,-1])

            a_trans, c_trans  = sess.run([accuracy, cross_entropy ], {X: test_features2_trans, Y_:  test_labels2_trans, tst: True})

            return_test_cost_trans(c_trans)
            return_test_accuracy_trans(a_trans,i,testEvery_trans)
  
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i})

 
for i in range(epochs): training_step(i, i , i % testEvery == 0, i % validateEvery==0)


runfile('/Users.../Phyton/plotDLs.py', wdir='/Users/.../Phyton')
 
mainTitle2='4.1_batchnorm_five_layers_relu--' + 'TransformTrainingData:' + str(TransormTrainingData
) +'.svg'

mainTitle='BN_5L_ReLU'+ '   ******* Translate: ' + str(translateImage
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
