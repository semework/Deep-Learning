#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:13:57 2017

@author: mulugetasemework
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:34:06 2017

@author: mulugetasemework


This code does geometric transfomations to "corrupt" data and increase 
the size of training and test sets. It is called by "processDataAndStup.py")
"""


import cv2
 

import numpy as np
 
import matplotlib.pyplot as plt   

from skimage.transform import warp

plt.close("all")

fig, ax = plt.subplots()

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'small',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small' }

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)

trX,trY = 1, 2

pylab.rcParams.update(params)

def TransformInputsDef(inputMatrix,in_labels,imageSize1,translateImage,rotateImage,rotationAngle,affineOrNot,perspectiveOrNot,WarpOrNot,keepDataSize):
    train_labels =  in_labels
    plt.subplot(231),plt.imshow(inputMatrix[-1],cmap='gray' ),plt.title('Input')
    
    if keepDataSize==1:
        d = list()
    else:
        d = list(inputMatrix)
        
    if translateImage==1:
        print("---  Translating " + str(len(inputMatrix)) + " images")
        if keepDataSize==0:
            train_labels = np.vstack([train_labels, train_labels])                     
          
        for im in range(len(inputMatrix)):
            img = inputMatrix[im]
            M = np.float32([[1,0,trX],[0,1,trY]])
            dst = cv2.warpAffine(img,M,(imageSize1,imageSize1))
            d.append(dst)
 

        plt.subplot(232),plt.imshow(dst,cmap='gray' ),plt.title('Translated', fontsize=7)
        plt.show()
       
 
    if rotateImage==1:
        print("----  rotating " + str(len(d)) + " images by: "  + str(rotationAngle) + " degrees")
        
        if keepDataSize==0:
            train_labels = np.vstack([train_labels, train_labels]) 

        d2=list() 
        for im in range(len(d)):
            img = d[im]
     
            M = cv2.getRotationMatrix2D((imageSize1/2,imageSize1/2),rotationAngle,1)
            
            dst = cv2.warpAffine(img,M,(imageSize1,imageSize1))
            if keepDataSize==1:
                d2.append(dst)
            else:
                d.append(dst)
        if keepDataSize==1:
            d=d2
            
        plt.subplot(233),plt.imshow(dst,cmap='gray' ),plt.title('Rotated', fontsize=7)
        plt.show()
 
    if affineOrNot==1:
        print("-----  Affine transforming  " + str(len(d)) + "  images...")
        if keepDataSize==0:
            train_labels = np.vstack([train_labels, train_labels])

        d2=list() 
        for im in range(len(d)):
            img = d[im]
            shift1 = 1
            pts1 = np.float32([[shift1,shift1],[shift1*4,shift1],[shift1,shift1*4]])
            pts2 = np.float32([[shift1/5,shift1*2],[shift1*4,shift1],[shift1*2,shift1*5]])
            
            M = cv2.getAffineTransform(pts1,pts2)
            
            dst = cv2.warpAffine(img,M,(imageSize1,imageSize1))

            if keepDataSize==1:
                d2.append(dst)

            else:
                d.append(dst)

        if keepDataSize==1:
            d=d2 

        plt.subplot(234),plt.imshow(dst,cmap='gray' ),plt.title('Affine-Transformed', fontsize=7)
        plt.show()

    if perspectiveOrNot==1:
        print("------  Perspective transforming  " + str(len(d)) + " images...")
        if keepDataSize==0:

           train_labels = np.vstack([train_labels, train_labels])

        d2=list() 
        for im in range(len(d)):
            img = d[im]
 
            pts1 = np.float32([[2,3],[imageSize1+1,4],[2,imageSize1+2],[imageSize1+3,imageSize1+4]])
            pts2 = np.float32([[0,0],[imageSize1-2,0],[0,imageSize1-2],[imageSize1-2,imageSize1-2]])
            
            M = cv2.getPerspectiveTransform(pts1,pts2)
            
            dst = cv2.warpPerspective(img,M,(imageSize1-2,imageSize1-2))
            dst =  np.resize(dst,[imageSize1,imageSize1])

            
            if keepDataSize==1:
                d2.append(dst)
            else:
                d.append(dst)
        if keepDataSize==1:
            d=d2            
        plt.subplot(235),plt.imshow(dst,cmap='gray' ),plt.title('Perspective-Transformed', fontsize=7)
        plt.show()
  
    if WarpOrNot==1:
        print("-------  Warping  " + str(len(d)) + " images...")
        if keepDataSize==0:
            train_labels = np.vstack([train_labels, train_labels])

        d2=list() 
        for im in range(len(d)):
            img = d[im]

            img *= 1/np.max(img)
            matrix = np.array([[1, 0, 0], [0, 1, -5], [0, 0, 1]])
 
            dst = warp(img, matrix)
            
            if keepDataSize==1:
                d2.append(dst)
            else:
                d.append(dst)
        if keepDataSize==1:
            d=d2

        plt.subplot(236),plt.imshow(dst,cmap='gray' ),plt.title('Warped', fontsize=7)
 
        plt.show()
 
    print("training label size :" + str(train_labels.shape))
    
    print("traiing data length :" + str(len(d)))
    return d,train_labels

 