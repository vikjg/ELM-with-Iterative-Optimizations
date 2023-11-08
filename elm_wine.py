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

tracemalloc.start()
model = randomNet(markers.size()[1], 500, wine.size()[1], torch.nn.functional.sigmoid)
elm = classifierELM(model, markers, wine, test_markers, test_wine)
init = elm.fit('element_gaussSeidel')
elm.classify()
print('Current and Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()