#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:08:57 2020

@author: Charlie, Chetana, Jonathan, Ashna, Elaine
"""


from continuous_network import ContinuousHopfieldNetwork, load_patterns
import logging
from typing import Optional
import numpy as np
from hopfield_base import HopfieldNetwork
import matplotlib.pyplot as plt
from mnistutils import load_mnist
from utils import *
from whitening import *

def prune (net, density=0.02, whiten=False):
    weights = net.weights.flatten()
    K = len(weights)
    ixs = np.random.choice(K, round(K * (1 - density)))
    weights[ixs] = 0
    weights = weights.reshape(net._N, net._N)
    if whiten:
        weights *= net._N * 2
    net._weights = weights

def loadPatterns (net, numPatterns, patternIndex, whitenVal, pattern='random'):
    # Create the network, and specify parameters.
    n_patterns = numPatterns
    ind_state_0 = patternIndex
    whiten = whitenVal

    # Load patterns, initialize transform, and train network.
    
    if pattern == 'written':
        P = load_patterns(n_patterns)
        P = zero_center(P, 1)
    else:
        P = np.random.rand(n_patterns, 28*28) * 2 - 1
    P0 = P.copy()
    tform = WhitenTransform("zca") if whiten else IdentityTransform()
    tform.fit(P)
    P = tform(P)
    net.store_patterns(P)
    
    return P0, ind_state_0, tform

def simulate(net, pattern = "random", 
             pauseVal=0.2, 
             numPatterns=5, 
             patternIndex=0, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             density=0.02):
    
    # Setup logging and some constants.
    logging.basicConfig(level=logging.INFO)

    # Create the network, and specify parameters.
    ind_state_0 = patternIndex
    
    P0, ind_state_0, tform = loadPatterns(net, numPatterns, patternIndex, whitenVal, pattern=pattern)
    prune(net, density=density, whiten=whitenVal)

    # Initialize state vector.
    state_0 = P0[ind_state_0]
    state_0_noisy = shuffle_degrade(state_0, 0.1)
    state_0_noisy = noise_degrade(state_0_noisy, 0.1)
    net.state = tform(state_0_noisy)

    # Setup plotting.
    if plotVal:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(state_0.reshape(28,28), cmap="gray")
        ax1.set_title("template")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(state_0_noisy.reshape(28, 28), cmap="gray")
        ax2.set_title("start")

        ax3 = fig.add_subplot(1, 3, 3)
    
    cdata = np.zeros([28, 28, 4])
    cdata[:, :, -1] = 1
    
    if plotVal:
        cimage = ax3.imshow(cdata)
        ax3.set_xticks([])
        ax3.set_yticks([])



    def update_plot(net: HopfieldNetwork) -> None:

        state = tform.inv(net.state)
        state = (state - state.min()) / (state.max() - state.min())
        state = state.reshape(28, 28)
        cdata[:, :, 0:3] = np.expand_dims(state, 2)
        
        if plotVal:
            cimage.set_data(cdata)
            ax3.set_title(f"iteration: {net.iteration}")
            if pauseVal>0:
                plt.pause(pauseVal)


    net.update_callback = update_plot
    net.record = ["state", "I_syn"]
    net.tau = tauVal
    net.run(itVal)
    
    return state_0.reshape(28, 28), cdata[0:28, 0:28, 0]

    
def calcPerf(templateMemory, 
             networkRecall, 
             n_rows, 
             n_cols, 
             plotVal=False, 
             arrayVal=False, 
             array=[]): 
    
    newTemplateMemory = templateMemory
    p = np.min(templateMemory) + 1    
    
    for x in range(n_rows):
        for y in range(n_cols):
            newTemplateMemory[x,y] = templateMemory[x,y] - p
        
    NNR = networkRecall
        
    for x in range(n_rows):
        for y in range(n_cols):
            a = networkRecall[x,y]
            NNR[x,y] = a*2 - 1
        
    difference = np.zeros((n_rows,n_cols))
    plotDifference = difference
        
    for x in range(n_rows):
        for y in range(n_rows):
            a = NNR[x,y]
            b = newTemplateMemory[x,y]
            difference[x,y] = abs(a-b)
            if difference[x,y] >= 1:
                print("(" + str(x) + ", " + str(y) + ")")
            if arrayVal:
                if array[x,y] == 1:
                    difference[x,y] = 0
                    plotDifference[x,y] = 1
                
    totalError = np.sum(difference)
    
    if not arrayVal:
        averageError = totalError/784
    else:
        averageError = totalError/(784 - np.sum(array))
    
    print("Average Error = " + str(format(averageError, '0.2f')))
    
    if plotVal:    
        plt.figure()
        plt.imshow(plotDifference, cmap = 'gray', vmin = 0, vmax = 2)
    
def measureWeights(net):
    numConnections = 0
    for x in range (784):
        for y in range (784):
            if net.weights[x,y] != 0:
                numConnections += 1
    return numConnections

def measureAverage(net):
    totalConnectionVal = 0
    for x in range (784):
        for y in range (784):
            totalConnectionVal += abs(net.weights[x,y])
    return totalConnectionVal/(784*784)

def measureAverageOfExistent(net):
    totalConnectionVal = 0
    numAlive = 0
    for x in range (784):
        for y in range (784):
            totalConnectionVal += abs(net.weights[x,y])
            if net.weights[x,y] != 0:
                numAlive += 1
    return totalConnectionVal/numAlive

def TBI (net, cod):
    array = np.zeros((1, 784))
    for i in range(784):
        chance = np.random.rand()
        if chance <= cod:
            array[0, i] = 1
            net.weights[:, i] = 0
            net.weights[i, :] = 0
    array = array.reshape(28, 28)
    return array
