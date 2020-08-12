#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:24:58 2020

@author: Charlie
"""
from continuous_network import ContinuousHopfieldNetwork
import numpy as np
from functionDefinitions import *
import matplotlib.pyplot as plt

n_rows, n_cols = 28, 28
tc = 784**2 - 784

def calculate (dens):

    network1 = ContinuousHopfieldNetwork(784)

    loadAndLearn(net=network1, numPatterns=5, patternIndex=2, whitenVal=False)

    a, b = simulate(net=network1, 
             pauseVal=0, 
             pattern='random', 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             density=dens)

    return calcPerf(a, b, 28, 28, plotVal=False)[0]

errorValues = np.zeros(25)
xVals = np.zeros(25)
    
for x in range(25):
    newDensity = x * 0.04
    xVals[x] = newDensity
    errorValues[x] = calculate(newDensity)