#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:27 2020

@author: Charlie, Jonathan, Chetana, Ashna, Elaine
"""


from continuous_network import ContinuousHopfieldNetwork
import numpy as np
from functionDefinitions import calcPerf, simulate, TBI, measureWeights
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
## set up parameters

R = 50
averageNum = 10
TBISeverity = 0.1
startDensity = 0.1
densityStep = 0.005

n_rows, n_cols = 28, 28
N = n_rows * n_cols

## Create the control net

baselineNet = ContinuousHopfieldNetwork(N)
baselineError = np.zeros(averageNum)
baselineDensity = np.zeros(averageNum)

## Simulate the control net

for x in range(averageNum):                  #clone, simulate, and store values
    clone = baselineNet
    a, b = simulate(net=baselineNet, 
                 pauseVal=0, 
                 numPatterns=5, 
                 patternIndex=2, 
                 whitenVal=False, 
                 plotVal=False, 
                 tauVal=0.1, 
                 itVal=50,
                 density=startDensity)
    
    baselineError[x] = calcPerf(a, b, 28, 28, plotVal=False)[0]
    baselineDensity[x] = measureWeights(clone)/(784**2 - 784)

baselineError = np.sum(baselineError)/averageNum           #average values
baselineDensity = np.sum(baselineDensity)/averageNum       #average values

print("Baseline Error = " + str(baselineError))
print("Baseline Density = " + str(baselineDensity))

## Create the TBI Net

TBINet = ContinuousHopfieldNetwork(N)
deadCells, deadConnections = TBI(TBINet, TBISeverity)

NDC = np.sum(deadCells)     #number of dead cells
NAC = 784 - NDC             #number of alive cells

TBIError = np.zeros(averageNum)
TBIDensity = np.zeros(averageNum)

## Simulate the TBI Net

for x in range(averageNum):
    clone = TBINet
    a, b = simulate(net=clone, 
                 pauseVal=0, 
                 numPatterns=5, 
                 patternIndex=2, 
                 whitenVal=False, 
                 plotVal=False, 
                 tauVal=0.1, 
                 itVal=50,
                 density = startDensity,
                 deadConnections=deadConnections)
    
    TBIError[x] = calcPerf(a, b, 28, 28, plotVal=False, arrayVal = True, array = deadCells)[0]
    TBIDensity[x] = measureWeights(clone)/(NAC**2 - NAC)

TBIError = np.sum(TBIError)/averageNum
TBIDensity = np.sum(TBIDensity)/averageNum

fTBIDensity = float(format(TBIDensity, '0.5f'))

print("TBI Error = " + str(TBIError))
print("TBI Density = " + str(TBIDensity))

## Simulate the healed nets

densityIncrease = [i * densityStep for i in range(R)]

errorArray = np.zeros(R)
minError = np.zeros(R)
maxError = np.zeros(R)
percentileMin = np.zeros(R)
percentileMax = np.zeros(R)

iterNum = 0
for i in densityIncrease:
    NEA = np.zeros(averageNum)
    D = i + fTBIDensity
    for x in range(averageNum): 
        clone = TBINet
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
    minError[iterNum] = np.min(NEA)
    percentileMin[iterNum] = np.percentile(NEA, 2.5)
    percentileMax[iterNum] = np.percentile(NEA, 97.5)
    maxError[iterNum] = np.max(NEA)
    print("Iteration: " + str(iterNum))
    iterNum += 1 

totalDensity = [(i + TBIDensity) for i in densityIncrease]

#def plotData():
plt.figure()
plt.plot(totalDensity, errorArray, 'r', label = 'After healing')
plt.plot(totalDensity, np.full(R, TBIError), 'k--', label='After TBI, 10% synaptic density')
plt.plot(totalDensity, np.full(R, baselineError), 'b--', label='Before TBI, 10% synaptic density')
plt.fill_between(totalDensity, percentileMin, percentileMax)
plt.ylim([0.16, 0.35])
plt.xlim([0.35, 0.10])
plt.xlabel("Synaptic Density (Proxy for Age)")
plt.ylabel("Error")
plt.legend()
