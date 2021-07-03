# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:13:11 2020

@author: Atakan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N =10000
d=10
toplam =0
secilenler=[]

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    toplam += odul

plt.hist(secilenler)
plt.show()

