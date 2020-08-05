#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:08:57 2020

@author: Charlie
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

def simulate(pauseVal=0.2, numPatterns=5, patternIndex=0, whitenVal=False, pruneVal=0.98, plotVal=False, tauVal=0.1, itVal=50):
    if __name__ == "__main__":
    
        # Setup logging and some constants.
        logging.basicConfig(level=logging.INFO)
        n_rows, n_cols = 28, 28
        N = n_rows * n_cols
    
        # Create the network, and specify parameters.
        net = ContinuousHopfieldNetwork(N)
        n_patterns = numPatterns
        ind_state_0 = patternIndex
        whiten = whitenVal
        prune = pruneVal
    
        # Load patterns, initialize transform, and train network.
        P = load_patterns(n_patterns)
        P = zero_center(P, 1)
        P0 = P.copy()
        tform = WhitenTransform("zca") if whiten else IdentityTransform()
        tform.fit(P)
        P = tform(P)
        net.store_patterns(P)
    
        weights = net.weights.flatten()
        K = len(weights)
        ixs = np.random.choice(K, round(K * prune))
        weights[ixs] = 0
        weights = weights.reshape(N, N)
        if whiten:
            weights *= N * 2
        net._weights = weights
    
    
        # Initialize state vector.
        state_0 = P0[ind_state_0]
        state_0_noisy = shuffle_degrade(state_0, 0.1)
        state_0_noisy = noise_degrade(state_0_noisy, 0.1)
        net.state = tform(state_0_noisy)
    
        # Setup plotting.
        if plotVal:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(state_0.reshape(n_rows, n_cols), cmap="gray")
            ax1.set_title("template")
    
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(state_0_noisy.reshape(n_rows, n_cols), cmap="gray")
            ax2.set_title("start")
    
            ax3 = fig.add_subplot(1, 3, 3)
        
        cdata = np.zeros([n_rows, n_cols, 4])
        cdata[:, :, -1] = 1
        
        if plotVal:
            cimage = ax3.imshow(cdata)
            ax3.set_xticks([])
            ax3.set_yticks([])
    
    
        def update_plot(net: HopfieldNetwork) -> None:
    
            state = tform.inv(net.state)
            state = (state - state.min()) / (state.max() - state.min())
            state = state.reshape(n_cols, n_rows)
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
        
        return state_0.reshape(n_rows, n_cols), cdata[0:28, 0:28, 0], n_rows, n_cols
    
##============ OUR CODE

    
def calcPerf(templateMemory, networkRecall, n_rows, n_cols, plotVal=False): 
    
    newTemplateMemory = templateMemory
        
    for x in range(n_rows):
        for y in range(n_cols):
            newTemplateMemory[x,y] = templateMemory[x,y] - (2/3)
        
    NNR = networkRecall
        
    for x in range(n_rows):
        for y in range(n_cols):
            a = networkRecall[x,y]
            NNR[x,y] = a*2 - 1
        
    difference = np.zeros((n_rows,n_cols))
        
    for x in range(n_rows):
        for y in range(n_rows):
            a = NNR[x,y]
            b = newTemplateMemory[x,y]
            difference[x,y] = abs(a-b)
            if difference[x,y] > 0.3:
                print("(" + str(x) + ", " + str(y) + ")")
                
    totalError = np.sum(difference)
    averageError = totalError/784
    
    print("Average Error = " + str(format(averageError, '0.2f')))
    
    if plotVal:    
        plt.figure()
        plt.imshow(difference, cmap = 'gray', vmin = 0, vmax = 2)
    

a, b, c, d = simulate(plotVal=False, patternIndex = 1, pauseVal = 0)
calcPerf(a, b, c, d, plotVal=False)
