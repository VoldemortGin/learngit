#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:09:36 2017

@author: gin
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [i*2 for i in x_data]

def forward(x):
    return x * w
    
def loss(x, y):
    return (y_pred - y)**2

w_list = []
mse_list = []

for w in np.arange(0., 4.1, .1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print('\t', x_val, y_val, y_pred, l)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
    
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('W')
plt.show()