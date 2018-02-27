#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:29:29 2017

@author: gin
"""

import torch
from torch import nn
from torch.autograd import Variable

x_data = [1., 2., 3.]
y_data = [i*2 for i in x_data]

w = Variable(torch.Tensor([1.]), requires_grad=True) # Any random value

                          