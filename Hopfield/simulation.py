#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:27 2020

@author: Charlie, Jonathan, Chetana, Ashna, Elaine
"""


from continuous_network import ContinuousHopfieldNetwork
import numpy as np
from functionDefinitions import calcPerf, simulate, prune, TBI, measureAverage, measureAverageOfExistent
from functionDefinitions import measureWeights

"""import time
seed = round(time.time())
np.random.seed(seed)"""

n_rows, n_cols = 28, 28
N = n_rows * n_cols
network1 = ContinuousHopfieldNetwork(N)

a, b = simulate(net=network1, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50,
             density=0.01)

calcPerf(a, b, 28, 28, plotVal=False)

beforeDensity = measureWeights(network1)/(784*784)

array1 = TBI(network1, 0.1)
aliveNumber = 784 - np.sum(array1)
aliveWeights = aliveNumber ** 2

a, b = simulate(net=network1, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=False, 
             tauVal=0.1, 
             itVal=50)

calcPerf(a, b, 28, 28, plotVal=False, arrayVal = True, array = array1)

afterNumber = measureWeights(network1)
afterDensity = afterNumber/aliveWeights

simulatedBeforeNum = aliveWeights * beforeDensity


'''array2 = array1.reshape(1, 784)

for y in range (784):
    if array2[0, y] == 1:
        for x in range(784):
            if network1.weights[x,y] != 0:
                print("error")'''


