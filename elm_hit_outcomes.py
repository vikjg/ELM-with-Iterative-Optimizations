import pandas as pd
#import numpy as np
import torch
#import torchvision
#import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
import tracemalloc
from extreme_learning_machines import randomNet, classifierELM
from sklearn.preprocessing import OneHotEncoder
#import pybaseball
  
# fetch dataset 
mlb = pd.read_csv(r'C:\Users\Vik/bip_of_outcomes.csv')

mlb = mlb.loc[mlb['events'].isin(['single', 'double', 'triple','home_run', 'field_out'])]

# data (as pandas dataframes) 
X = mlb[['launch_speed', 'launch_angle', 'spray_angle']]
y = mlb[['events']]


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
model = randomNet(markers.size()[1], 500, pitch.size()[1], torch.nn.functional.sigmoid)
elm = classifierELM(model, markers, pitch, test_markers, test_pitch)
init = elm.fit('element_gaussSeidel')
elm.classify()
print('Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()
