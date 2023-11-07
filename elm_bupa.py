# From metadata: " The 7th field was created by BUPA researchers as a train/test selector. 
# It is not suitable as a dependent variable for classification. The dataset does not contain 
# any variable representing presence or absence of a liver disorder."
# As such, this will be used as a test of the regressionELM to approximate drinks per day.

import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from extreme_learning_machines import randomNet, regressionELM, to_onehot
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
liver_disorders = fetch_ucirepo(id=60) 
  
# data (as pandas dataframes) 
X = liver_disorders.data.features 
y = liver_disorders.data.targets 
  
features = torch.from_numpy(pd.DataFrame(X).to_numpy()).type(torch.float)
targets = torch.from_numpy(pd.DataFrame(y).to_numpy()).type(torch.float)

model = randomNet(5, 1000, 1, torch.nn.functional.sigmoid)
elm = regressionELM(model, features, targets, features[21:29], targets[21:29])
init = elm.fit('pseudo_inv')
print(elm.classify())
print(targets[21:29])
#print(features[27:29].view(features[27:29].size(0),-1).size())

