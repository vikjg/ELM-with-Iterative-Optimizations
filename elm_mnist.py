import pandas as pd
import numpy as np
import torch
#import torchvision
#import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import tracemalloc
from extreme_learning_machines import randomNet, classifierELM

# One hot encoder for the MNIST targets
def to_onehot(batch_size, num_classes, y):
    y_onehot = torch.FloatTensor(batch_size, num_classes)
    y = torch.unsqueeze(y, dim=1)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('C:/Users/Vik/ELM-with-Iterative-Optimizations/.data', download=True, train=True, transform=transform)
valset = datasets.MNIST('C:/Users/Vik/ELM-with-Iterative-Optimizations/.data', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=True)


dataiter = iter(trainloader)
images, labels = next(dataiter)
labels = to_onehot(batch_size=len(labels), num_classes=10, y=labels)

test_dataiter = iter(valloader)
test_images, test_labels = next(test_dataiter)
test_labels = to_onehot(batch_size=len(test_labels), num_classes=10, y=test_labels)

# =============================================================================
# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
# 
# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# =============================================================================

tracemalloc.start()
model = randomNet(784, 500, 10, torch.nn.functional.sigmoid)
elm = classifierELM(model, images, labels, test_images, test_labels, 500, 1e-5)
tracemalloc.start()
init, train_t = elm.fit('element_gaussSeidel')
print('Peak RAM Usage:', tracemalloc.get_traced_memory()[1]/1000000, 'MB')
tracemalloc.stop()
acc, test_t = elm.classify()


# =============================================================================
# optimizers = ['element_gaussSeidel', 'element_jacobi', 'SOR', 'pseudo_inv', 'pinv']
# neuron = [10, 100, 500]
# tols = [1e-1, 1e-3, 1e-5]
# activation = [torch.nn.functional.sigmoid, torch.nn.functional.tanh, torch.sin]
# max_iter = [50, 100, 500]
# results = pd.DataFrame(columns=['Optimizer', 'Neurons', 'Activation Func', 'Max Iterations', 
#                                  'Training Time (s)', 'Testing Time (s)', 'RAM Usage (MB)', 'Testing Accuracy'])
# opt_dict = {'element_gaussSeidel': 'GS', 'element_jacobi': 'Jac', 'SOR': 'SOR', 'pseudo_inv':'MP Psuedo-Inv Built-In', 'pinv':'MP Psuedo-Inv Hard Code'}
# act_dict = {torch.nn.functional.sigmoid:'Sigmoid', torch.nn.functional.tanh:'tanh', torch.sin:'sin'}
# 
# for opt in optimizers:
#     for neurons in neuron:
#         for activations in activation:
#             for max_iters in max_iter:
#                 for tol in tols:
#                     tracemalloc.start()
#                     model = randomNet(784, neurons, labels.size()[1], activations)
#                     elm = classifierELM(model, images, labels, test_images, test_labels, max_iters, tol)
#                     init, train_t = elm.fit(opt)
#                     acc, test_t = elm.classify()
#                     peak_ram = tracemalloc.get_traced_memory()[1]/1000000
#                     tracemalloc.stop()
#                     
#                     new_row = {'Optimizer':opt_dict[opt], 'Neurons':neurons,'Activation Func': act_dict[activations], 'Max Iterations':max_iters,
#                                'Tolerance':tol, 'Training Time (s)':train_t, 'Testing Time (s)':test_t, 
#                                'RAM Usage (MB)':peak_ram, 'Testing Accuracy':acc}
#                     results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
# =============================================================================
                


