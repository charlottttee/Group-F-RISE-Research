#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:27 2020

@author: Charlie, Jonathan, Chetana, Ashna, Elaine
"""


from continuous_network import ContinuousHopfieldNetwork
import numpy as np
from functionDefinitions import calcPerf, simulate, TBI
import matplotlib.pyplot as plt
##from functionDefinitions import measureWeights

"""import time
seed = round(time.time())
np.random.seed(seed)"""

def cloneAndSimulate (copyNet, density):
    net = copyNet
    a, b = simulate(net=net,
             pauseVal = 0, 
             numPatterns = 5, 
             patternIndex = 2, 
             whitenVal = False, 
             plotVal = False, 
             tauVal = 0.1, 
             itVal = 50, 
             density = density)
    error = calcPerf(a, b, 28, 28, False)
    return error

n_rows, n_cols = 28, 28
N = n_rows * n_cols

## Create the control net

baselineNet = ContinuousHopfieldNetwork(N)

## Simulate the control net

a, b = simulate(net=baselineNet, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             density=0.02)

calcPerf(a, b, 28, 28, plotVal=False)

## Create the TBI Net

TBINet = ContinuousHopfieldNetwork(N)
deadCells, deadConnections = TBI(TBINet, 0.1)

densityList = [0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07]
errorArray = np.zeros(10)

for i in densityList:
    iterNum = 0
    errorArray[iterNum] = cloneAndSimulate(TBINet, i)

## Simulate the TBI Net

a, b = simulate(net=TBINet, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             deadConnections=deadConnections)

calcPerf(a, b, 28, 28, plotVal=False, arrayVal = True, array = deadCells)

##afterNumber = measureWeights(TBINet)
##afterDensity = afterNumber/aliveWeights
##simulatedBeforeNum = aliveWeights * beforeDensity

