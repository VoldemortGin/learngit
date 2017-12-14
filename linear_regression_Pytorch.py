# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.], [2.], [3.]]))
y_data = Variable(torch.Tensor([[2.], [4.], [6.]]))

#==============================================================================
# Design your model using class
#==============================================================================

class Model(torch.nn.Module):
    def __init__(self):
        # In the constructor we instantiate two nn.Linear Module
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in and one out
        
    def forward(self, x):
        '''
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        '''
        y_pred = self.linear(x)
        return y_pred

# our model
model = Model()

#==============================================================================
# construct loss and optimizer
#==============================================================================

#construct our loss function and an Optimizer. The call to model.parameters()
#in the SGD constructor will contain the learnable parameters of the two
#nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#==============================================================================
# Training cycle (forward, backward, update)
#==============================================================================

# Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
    
    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
hour_var = Variable(torch.Tensor([[4.]]))
print('predict (after training)', 4, model.forward(hour_var).data[0][0])