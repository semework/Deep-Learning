#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:52:27 2017

@author: mulugetasemework

This code is called by neural net models at the end of their run.
"""

import matplotlib.pyplot as plt  

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

ax1.plot(trIndx[:-1],[i * 100 for i in trAcc],'b--',label='training accuracy',ms=2)

testAccS[0] = 0
if test_thiscode==1: 

    ax1.plot(testIndxS,[i * 100 for i in testAccS],':',label='swapped test accuracy')
 
ax1.plot(testIndx_trans,[i * 100 for i in testAcc_trans],'m*-',label='trans_test accuracy',linewidth=1.5)    
ax1.plot(testIndx,[i * 100 for i in testAcc],'r',label='test accuracy',linewidth=1.5)
if test_shuffled == 1:

    ax1.plot(testIndx_shuff,[i * 100 for i in testAcc_shuff],'k-.',label='shuffled test accuracy')

if doNotValidate==0:
    ax1.plot(validIndx,[i * 100 for i in validAcc],'green',label='validation accuracy')

ax1.set_title('Accuracy',size=8)
ax1.set_xlabel('Epochs',size=6)
ax1.set_ylabel('Percent correct',size=6)
ax1.set_ylim(-1, 105)

labels=ax1.get_xticks()+1
labels=[int(x) for x in labels]

ax1.set_xticklabels(labels)

if test_shuffled == 1:
    maxY=max(max(testCostS),max(trCost),max(testCost),max(testCost_trans),max(testCost_shuff))
else:
    maxY=max(max(testCostS),max(trCost),max(testCost),max(testCost_trans))
    
newMax = maxY;

if test_thiscode==1:
    if newMax > 100:
        ax2.plot(testIndxS,[(i/newMax)*100 for i in testCostS] ,':',label='swapped_test')
        if test_shuffled == 1:
            ax2.plot(testIndxS,[(i/newMax)*100 for i in testCost_shuff] ,'k-.',label='shuffled test')            
        ax2.plot(trIndx[:-1], [(i/newMax)*100 for i in trCost],'b--',label='training',ms=2.5)

        ax2.plot(testIndx_trans,[(i/newMax)*100 for i in testCost_trans],'m*-',label='trans_test',linewidth=1.5)
        ax2.plot(testIndx,[(i/newMax)*100 for i in testCost],'r',label='test',linewidth=1.5)
        
    else:
        ax2.plot(testIndxS, testCostS ,':',label='swapped_test')
        if test_shuffled == 1:
            ax2.plot(testIndx_shuff,testCost_shuff ,'k-.',label='shuffled test')   
        ax2.plot(trIndx[:-1],  trCost,'b--',label='training',ms=2.5)

        ax2.plot(testIndx_trans, testCost_trans,'m*-',label='trans_test',linewidth=1.5)
        ax2.plot(testIndx, testCost,'r',label='test',linewidth=1.5)


if doNotValidate==0:
    ax2.plot(validIndx, validCost ,'green',label='validation')

ax2.set_xticklabels(  (labels))
ax2.set_title('Cost' ,size=8)
ax2.set_xlabel('Epochs',size=6)
ax2.set_ylabel('Entropy',size=6 )


ax2.set_ylim(-1, 105)
ax2.legend(loc='upper right',fontsize=6)

zed = [tick.label.set_fontsize(5) for tick in ax1.xaxis.get_major_ticks()]
zed = [tick.label.set_fontsize(5) for tick in ax1.yaxis.get_major_ticks()]
zed = [tick.label.set_fontsize(5) for tick in ax2.xaxis.get_major_ticks()]
zed = [tick.label.set_fontsize(5) for tick in ax2.yaxis.get_major_ticks()]
