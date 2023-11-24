import pandas as pd
#import numpy as np
import torch
#import torchvision
#import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
import tracemalloc
from extreme_learning_machines import randomNet, classifierELM
from sklearn.preprocessing import OneHotEncoder
  
# fetch dataset 
mlb = pd.read_csv(r'C:\Users\vgiorda1\Python\ELM-with-Iterative-Optimizations\2023_mlb_statcast.csv')
  
# data (as pandas dataframes) 
X = mlb[['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 'spin_axis']]
y = mlb[['pitch_type']]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y_onehot = enc.transform(y).toarray()


features = torch.from_numpy(pd.DataFrame(X).to_numpy(dtype=float)).type(torch.float)
targets = torch.from_numpy(y_onehot).type(torch.float)

tensor_dataset = torch.utils.data.TensorDataset(features, targets)
trainloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)
valloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=len(tensor_dataset)//3, shuffle=True)

dataiter = iter(trainloader)
markers, pitch = next(dataiter)


test_dataiter = iter(valloader)
test_markers, test_pitch = next(test_dataiter)

tracemalloc.start()
model = randomNet(markers.size()[1], 200, pitch.size()[1], torch.nn.functional.sigmoid)
elm = classifierELM(model, markers, pitch, test_markers, test_pitch)
init = elm.fit('element_gaussSeidel')
elm.classify()
print('Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()