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
model = randomNet(markers.size()[1], 200, pitch.size()[1], torch.nn.functional.sigmoid)
elm = classifierELM(model, markers, pitch, test_markers, test_pitch, 100, 1e-5)
init, train_t = elm.fit('pinv')
acc, test_t = elm.classify()
print('Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()


# =============================================================================
# optimizers = ['element_gaussSeidel', 'element_jacobi', 'SOR', 'pseudo_inv']
# neuron = [10, 50, 100, 500]
# activation = [torch.nn.functional.sigmoid, torch.nn.functional.relu, torch.nn.functional.tanh, torch.sin]
# max_iter = [50, 100, 500]
# results = pd.DataFrame(columns=['Optimizer', 'Neurons', 'Activation Func', 'Max Iterations', 
#                                  'Training Time (s)', 'Testing Time (s)', 'RAM Usage (MB)', 'Testing Accuracy'])
# opt_dict = {'element_gaussSeidel': 'GS', 'element_jacobi': 'Jac', 'SOR': 'SOR', 'pseudo_inv':'MP Psuedo-Inv'}
# act_dict = {torch.nn.functional.sigmoid:'Sigmoid', torch.nn.functional.relu:'RELU', torch.nn.functional.tanh:'tanh', torch.sin:'sin'}
# 
# for opt in optimizers:
#     for neurons in neuron:
#         for activations in activation:
#             for max_iters in max_iter:
#                 tracemalloc.start()
#                 model = randomNet(markers.size()[1], neurons, pitch.size()[1], activations)
#                 elm = classifierELM(model, markers, pitch, test_markers, test_pitch, max_iters)
#                 init, train_t = elm.fit(opt)
#                 acc, test_t = elm.classify()
#                 peak_ram = tracemalloc.get_traced_memory()[1]/1000000
#                 tracemalloc.stop()
#                 
#                 new_row = {'Optimizer':opt_dict[opt], 'Neurons':neurons,'Activation Func': act_dict[activations], 'Max Iterations':max_iters,
#                            'Training Time (s)':train_t, 'Testing Time (s)':test_t, 'RAM Usage (MB)':peak_ram, 'Testing Accuracy':acc}
#                 results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
# =============================================================================

                