# From metadata: " The 7th field was created by BUPA researchers as a train/test selector. 
# It is not suitable as a dependent variable for classification. The dataset does not contain 
# any variable representing presence or absence of a liver disorder."
# As such, this will be used as a test of the regressionELM to approximate drinks per day.

import pandas as pd
#import numpy as np
import torch
#import torchvision
#import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
import tracemalloc
from extreme_learning_machines import randomNet, regressionELM
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
liver_disorders = fetch_ucirepo(id=60) 

# data (as pandas dataframes) 
X = liver_disorders.data.features 
y = liver_disorders.data.targets 
  
features = torch.from_numpy(pd.DataFrame(X).to_numpy()).type(torch.float)
targets = torch.from_numpy(pd.DataFrame(y).to_numpy()).type(torch.float)
tensor_dataset = torch.utils.data.TensorDataset(features, targets)
trainloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)
valloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)

dataiter = iter(trainloader)
markers, drinks = next(dataiter)


test_dataiter = iter(valloader)
test_markers, test_drinks = next(test_dataiter)



tracemalloc.start()
model = randomNet(markers.size()[1], 500, 1, torch.sin)
elm = regressionELM(model, markers, drinks, test_markers, test_drinks, 50)
init = elm.fit('pseudo_inv')
test = elm.classify()
print('RMSE:', test)
print('Current and Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()


