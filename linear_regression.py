#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:57:00 2017

@author: gin
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [i*2 for i in x_data]

w = 3.0 # any random value

# our model forward pass
def forward(x):
    return x*w
    
# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# compute gradient
def gradient(x, y):
    return 2 * x * (x * w - y)  # the derivative of loss.

# before training
print("predict (before training)", 4, '\t', forward(4.))

# Training loop
for epoch in range(10):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.1 * grad      # update w
        print('\t\tgrad:', x, y, grad)
        l = loss(x, y)
    
    print('progress:', epoch, l)

# After training
print('predict (after training)', 4, forward(4.))