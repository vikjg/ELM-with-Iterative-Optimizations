import pandas as pd
#import numpy as np
import torch
#import torchvision
#import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
import tracemalloc
from extreme_learning_machines import randomNet, regressionELM
from ucimlrepo import fetch_ucirepo 
  

abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
X = abalone.data.features 
X = pd.DataFrame(X).drop(columns=['Sex'])
y = abalone.data.targets

features = torch.from_numpy(pd.DataFrame(X).to_numpy()).type(torch.float)
targets = torch.from_numpy(pd.DataFrame(y).to_numpy()).type(torch.float)

tensor_dataset = torch.utils.data.TensorDataset(features, targets)

trainloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)
valloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)

dataiter = iter(trainloader)
markers, age = next(dataiter)


test_dataiter = iter(valloader)
test_markers, test_age = next(test_dataiter)

tracemalloc.start()
model = randomNet(markers.size()[1], 500, 1, torch.sin)
elm = regressionELM(model, markers, age, test_markers, test_age, 50)
init = elm.fit('element_gaussSeidel')
test = elm.classify()
print('RMSE:', test)
print('Current and Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()