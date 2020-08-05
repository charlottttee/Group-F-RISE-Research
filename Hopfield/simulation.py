#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:27 2020

@author: Charlie
"""


from continuous_network import ContinuousHopfieldNetwork
from functionDefinitions import calcPerf, simulate, prune, TBI

"""import time
seed = round(time.time())
np.random.seed(seed)"""

n_rows, n_cols = 28, 28
N = n_rows * n_cols
network1 = ContinuousHopfieldNetwork(N)

prune(network1)

array = TBI(network1, 0.1)

a, b = simulate(net=network1, 
             pauseVal=0, 
             numPatterns=5, 
             patternIndex=2, 
             whitenVal=False, 
             plotVal=True, 
             tauVal=0.1, 
             itVal=50)

''''calcPerf(a, b, 28, 28, plotVal=True)'''