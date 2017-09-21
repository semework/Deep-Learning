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

 

learning_rate=0.0000000001
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, imageSize1*imageSize1]        # imageSize1*imageSize1 = imageSize1 * imageSize1
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [imageSize1*imageSize1, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of imageSize1ximageSize1 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with imageSize1*imageSize1 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns



# input X: imageSize1ximageSize1 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, imageSize1,imageSize1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, n_classes])
# weights W[imageSize1*imageSize1, n_classes]   imageSize1*imageSize1=imageSize1*imageSize1
W = tf.Variable(tf.zeros([imageSize1*imageSize1, n_classes]))
# biases b[n_classes]
b = tf.Variable(tf.zeros([n_classes]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, imageSize1*imageSize1])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def training_step(i, update_train_data, update_test_data, update_valid_data):
 
    thisCountTr = return_counterUpdateTr()
    start = thisCountTr[-1]
    end =     start +  batch_size
    batch_X,batch_Y = train_features[start:end], train_labels[start:end]  

    # compute training values 
    if update_train_data:
        a, c,  w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X, Y_: batch_Y})
        print(str(i) + ": |--------- " + str(a) +   " --- " + str(c) +
              " --- <-Training accuracy:" +   " <- loss: "  + " : epoch " +
              str(i*100//len(train_features)+1)  + " (lr:" + str(learning_rate) + ")")
  
        return_train_cost(c)
        return_train_accuracy(a,i,testEvery) 
        if TransormTrainingData==1:
            if end <=len( test_features):
                batch_X_trans,batch_Y_trans = train_features_trans[start:end], train_labels_trans[start:end]  
                a_trans, c_trans,  w_trans, b_trans = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X_trans, Y_: batch_Y_trans})
    
                return_train_cost_trans(c_trans)
                return_train_accuracy_trans(a_trans) 
   
    if update_valid_data and doNotValidate == 0:
        startV = i
        end =   startV + 1
        batch_X_valid,batch_Y_valid = valid_features[startV:end], valid_labels[startV:end] 
        a, valid_cost,   w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X_valid, Y_: batch_Y_valid})
        print(str(i) + ":*** Validation accuracy:" + str(a) +
              " loss: " + str(valid_cost) + " (lr:" + str(learning_rate) + ")")
        return_valid_cost(valid_cost)
        return_valid_accuracy(a,i)   

    if update_test_data:
        thisCount = return_counterUpdate()
        startTst = thisCount[-1]
        end =     startTst  + test_batch_size
        if end <=len( test_features):
            batch_X_test,test_labels2 = test_features[startTst:end], test_labels[startTst:end] 
            a, c  = sess.run([accuracy, cross_entropy ], {X: batch_X_test, Y_: test_labels2})
            print(str(i) + ": ********* epoch " + str(i*100//len(test_features)+1) +
                  " ********* test accuracy:" + str(a) + " test loss: " + str(c))
      
            return_test_cost(c)
            return_test_accuracy(a,i)
          
            if test_thiscode==1:
                test_labels3 = np.array(swapped_test_labels[startTst:end])
                aS, ctestS= sess.run([accuracy, cross_entropy ], {X: batch_X_test, Y_: test_labels3})
                print("inside code test")
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
    
                aS_shuff, ctestS_shuff  = sess.run([accuracy, cross_entropy ], {X:  (batch_X_shuff), Y_:  (batch_Y_shuff) })
                
                return_test_cost_shuff(ctestS_shuff)
                return_test_accuracy_shuff(aS_shuff,i)     

        thisCount = return_counterUpdate_trans()
        startTst_trans = thisCount[-1]
        end_trans =     startTst_trans  + test_batch_size_trans
        if  end_trans <= len(test_features_trans): 
            batch_X_test_trans,test_labels2_trans = test_features_trans[startTst_trans:end_trans], test_labels_trans[startTst_trans: end_trans] 
            a_trans, c_trans  = sess.run([accuracy, cross_entropy ], {X: batch_X_test_trans, Y_: test_labels2_trans})
   
            return_test_cost_trans(c_trans)
            return_test_accuracy_trans(a_trans,i,testEvery_trans)     
        

    sess.run(train_step, {X: batch_X, Y_: batch_Y })


for i in range(epochs): training_step(i, i , i % testEvery == 0, i % validateEvery==0)


runfile('/Users/mulugetasemework/Dropbox/Phyton/plotDLs.py', wdir='/Users/mulugetasemework/Dropbox/Phyton')


mainTitle2='1L_softmax--' + 'TransformTrainingData:' + str(TransormTrainingData
) +'.svg'
                                                                    
mainTitle='1_layer_softmax '+ '   ******* Translate: ' + str(translateImage
 )+ '    Rotate: ' + str(rotateImage)+ '   Affine: ' + str(affineOrNot
)+ '   Perspective: ' + str(perspectiveOrNot)+ '   Warp: ' + str(WarpOrNot
) + '   keepDataLength:   ' + str(keepDataSize
) + '   TransformTrainingData:   ' + str(TransormTrainingData
) + ' \n  Learning_rate :   ' + str(learning_rate)

figDir="/Users/mulugetasemework/Documents/Python/"
 
figname= mainTitle+'.svg'

f.suptitle(mainTitle,size=7 ) 
plt.subplots_adjust(left=0.1, wspace=0.2, top=0.7, bottom=0.2)
f.show()
os.chdir(figDir)

#plt.savefig(mainTitle2, format='svg', dpi=1200)    
