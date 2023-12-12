import pandas as pd
#import numpy as np
import torch
#import torchvision
#import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
import tracemalloc
from extreme_learning_machines import randomNet, classifierELM
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder
  
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y_onehot = enc.transform(y).toarray()


features = torch.from_numpy(pd.DataFrame(X).to_numpy(dtype=float)).type(torch.float)
targets = torch.from_numpy(y_onehot).type(torch.float)

tensor_dataset = torch.utils.data.TensorDataset(features, targets)
trainloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)
valloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)

dataiter = iter(trainloader)
markers, wine = next(dataiter)


test_dataiter = iter(valloader)
test_markers, test_wine = next(test_dataiter)


# =============================================================================
# model = randomNet(markers.size()[1], 500, wine.size()[1], torch.nn.functional.tanh)
# 
# elm = classifierELM(model, markers, wine, test_markers, test_wine, 500, 1e-5)
# tracemalloc.start()
# init, train_t = elm.fit('element_gaussSeidel')
# print('Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
# tracemalloc.stop()
# acc, test_t = elm.classify()
# =============================================================================

optimizers = ['element_gaussSeidel', 'element_jacobi', 'SOR', 'pseudo_inv', 'pinv']
neuron = [10, 100, 500]
tols = [1e-1, 1e-3, 1e-5]
activation = [torch.nn.functional.sigmoid, torch.nn.functional.tanh, torch.sin]
max_iter = [50, 100, 500]
results = pd.DataFrame(columns=['Optimizer', 'Neurons', 'Activation Func', 'Max Iterations', 
                                 'Training Time (s)', 'Testing Time (s)', 'RAM Usage (MB)', 'Testing Accuracy'])
opt_dict = {'element_gaussSeidel': 'GS', 'element_jacobi': 'Jac', 'SOR': 'SOR', 'pseudo_inv':'MP Psuedo-Inv Built-In', 'pinv':'MP Psuedo-Inv Hard Code'}
act_dict = {torch.nn.functional.sigmoid:'Sigmoid', torch.nn.functional.tanh:'tanh', torch.sin:'sin'}
i = 0
for opt in optimizers:
    for neurons in neuron:
        for activations in activation:
            for max_iters in max_iter:
                for tol in tols:
                    tracemalloc.start()
                    model = randomNet(markers.size()[1], neurons, wine.size()[1], activations)
                    elm = classifierELM(model, markers, wine, test_markers, test_wine, max_iters, tol)
                    init, train_t = elm.fit(opt)
                    acc, test_t = elm.classify()
                    peak_ram = tracemalloc.get_traced_memory()[1]/1000000
                    tracemalloc.stop()
                    
                    new_row = {'Optimizer':opt_dict[opt], 'Neurons':neurons,'Activation Func': act_dict[activations], 'Max Iterations':max_iters,
                               'Tolerance':tol, 'Training Time (s)':train_t, 'Testing Time (s)':test_t, 
                               'RAM Usage (MB)':peak_ram, 'Testing Accuracy':acc}
                    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                    i+=1
                    if i % 10 == 0:
                        print(i)
                    