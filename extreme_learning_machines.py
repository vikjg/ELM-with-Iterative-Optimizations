
# =============================================================================
# import sys
# sys.path.append('C:/Users/vgiorda1/Python/ELM-with-Iterative-Optimizations')
# =============================================================================

import torch
import torch.nn as nn
#import torchvision
#import matplotlib.pyplot as plt
from time import time
from math import sqrt
#from torchvision import datasets, transforms
from optimizers import optimizer



class randomNet(nn.Module):
    def __init__(self, size_input, hidden_neurons, size_output, activation_func):
        self.size_input = size_input
        self.hidden_neurons = hidden_neurons   
        self.size_output = size_output
        self.activation_func = activation_func
        
        super(randomNet, self).__init__()
        self.layer1 = nn.Linear(size_input, hidden_neurons)
        if activation_func == torch.nn.functional.leaky_relu:
            torch.nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        if activation_func == torch.nn.functional.relu:
            torch.nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
        else:
            torch.nn.init.xavier_uniform_(self.layer1.weight, gain=1)
        self.layer2 = nn.Linear(hidden_neurons, size_output, bias=False)
        
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.layer1(x)
        x = self.activation_func(x)
        x = self.layer2(x)
        return x
    
    def forwardToHidden(self, x):
       x = x.view(x.size(0),-1)
       x = self.layer1(x)
       x = self.activation_func(x)
       return x

class classifierELM(): 
    def __init__(self, model, train_data, target, test_data, test_target):
        self.model = model
        self.train_data = train_data
        self.target = target
        self.test_data = test_data
        self.test_target = test_target

    
    def fit(self, optimizer_func):
        init_time = time()
        hidden = self.model.forwardToHidden(self.train_data)
        opt = optimizer(self.model, hidden, self.target, 500)
        beta = optimizer_call(opt, optimizer_func)
        with torch.no_grad():
            self.model.layer2.weight = torch.nn.parameter.Parameter(beta.t())
        output = self.model.forward(self.train_data)
        end_time = time()
        print('Training time:', end_time - init_time)
        return output
    
    def classify(self):
        init_time = time()
        output = self.model.forward(self.test_data)
        end_time = time()
        correct = torch.sum(torch.argmax(output, dim=1) == torch.argmax(self.test_target, dim=1)).item()
        print(correct / len(self.test_data))
        print('Test time:', end_time - init_time)
        

class regressionELM(): 
    def __init__(self, model, train_data, target, test_data, test_target):
        self.model = model
        self.train_data = train_data
        self.target = target
        self.test_data = test_data
        self.test_target = test_target
        
    
    def fit(self, optimizer_func):
        init_time = time()
        hidden = self.model.forwardToHidden(self.train_data)
        opt = optimizer(self.model, hidden, self.target, 500)
        beta = optimizer_call(opt, optimizer_func)
        with torch.no_grad():
            self.model.layer2.weight = torch.nn.parameter.Parameter(beta.t())
        output = self.model.forward(self.train_data)
        end_time = time()
        print('Training time:', end_time - init_time)
        return output
        
    def classify(self):
        init_time = time()
        output = self.model.forward(self.test_data)
        mse = nn.MSELoss()
        end_time = time()
        print('Test time:', end_time - init_time)
        return sqrt(mse(output, self.test_target))
    
# Helper Functions
def optimizer_call(optimizer, optimizer_func):
    if optimizer_func == 'pseudo_inv':    
        beta = optimizer.pseudo_inv()
    if optimizer_func == 'jacobi':
        beta = optimizer.jacobi()
    if optimizer_func == 'element_jacobi':
        beta = optimizer.element_jacobi()
    if optimizer_func == 'gaussSeidel':
        beta = optimizer.gaussSeidel()
    if optimizer_func == 'element_gaussSeidel':
        beta = optimizer.element_gaussSeidel()
    if optimizer_func == 'SOR':
        beta = optimizer.SOR()    
    
    return beta


        
        
