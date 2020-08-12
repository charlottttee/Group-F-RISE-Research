#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:27 2020

@author: Charlie, Jonathan, Chetana, Ashna, Elaine
"""


from continuous_network import ContinuousHopfieldNetwork
import numpy as np
from functionDefinitions import *
import matplotlib.pyplot as plt
##from functionDefinitions import measureWeights

"""import time
seed = round(time.time())
np.random.seed(seed)

def cloneAndSimulate (copyNet, den, DC):
    net = copyNet
    a, b = simulate(net=net,
             pauseVal = 0, 
             numPatterns = 5, 
             patternIndex = 2, 
             whitenVal = False, 
             plotVal = False, 
             tauVal = 0.1, 
             itVal = 50, 
             density = den, 
             deadConnections=DC)
    error = calcPerf(a, b, 28, 28, False)[0]
    return error"""

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
             density=0.1)

baselineError = calcPerf(a, b, 28, 28, plotVal=False)[0]

print("Baseline Error = " + str(baselineError))
print("Baseline Density = " + str(measureWeights(baselineNet)/(784**2 - 784)))

## Create the TBI Net

TBINet = ContinuousHopfieldNetwork(N)
deadCells, deadConnections = TBI(TBINet, 0.05)

TBITemplate = TBINet

NDC = np.sum(deadCells)
NAC = 784 - NDC

## Simulate the TBI Net

a, b = simulate(net=TBINet, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             density = 0.1,
             deadConnections=deadConnections)

TBIError = calcPerf(a, b, 28, 28, plotVal=False, arrayVal = True, array = deadCells)[0]
TBIDensity = measureWeights(TBINet)/(NAC**2 - NAC)
fTBIDensity = float(format(TBIDensity, '0.5f'))


print("TBI Error = " + str(TBIError))
print("TBI Density = " + str(TBIDensity))

## Simulate the healed nets

R = 50
averageNum = 10

densityIncrease = [i * 0.005 for i in range(R)]

errorArray = np.zeros(R)

iterNum = 0
for i in densityIncrease:
    NEA = np.zeros(averageNum)
    D = i + fTBIDensity
    for x in range(averageNum): 
        clone = TBITemplate
        c, t = simulate(net=clone, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             density = D,
             deadConnections=deadConnections)
        cloneError = calcPerf(c, t, 28, 28, plotVal=False, arrayVal = True, array = deadCells)[0]
        NEA[x] = cloneError
    errorArray[iterNum] = np.sum(NEA)/averageNum
    print("Iteration: " + str(iterNum))
    iterNum += 1

plt.figure()    
plt.plot(errorArray)
plt.plot(np.full(R, TBIError))
plt.plot(np.full(R, baselineError))

##afterNumber = measureWeights(TBINet)
##afterDensity = afterNumber/aliveWeights
##simulatedBeforeNum = aliveWeights * beforeDensity
