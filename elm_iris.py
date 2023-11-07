import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from extreme_learning_machines import randomNet, classifierELM, to_onehot
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y_onehot = onehotlabels = enc.transform(y).toarray()


features = torch.from_numpy(pd.DataFrame(X).to_numpy(dtype=float)).type(torch.float)
targets = torch.from_numpy(y_onehot).type(torch.float)

model = randomNet(features.size()[1], 500, targets.size()[1], torch.nn.functional.sigmoid)
elm = classifierELM(model, features, targets, features[1:10], targets[1:10])
init = elm.fit('element_gaussSeidel')
elm.classify()
