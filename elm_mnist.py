import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from extreme_learning_machines import randomNet, classifierELM



def to_onehot(batch_size, num_classes, y):
    y_onehot = torch.FloatTensor(batch_size, num_classes)
    y = torch.unsqueeze(y, dim=1)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('C:/Users/vgiorda1/Python/ELM-with-Iterative-Optimizations/.data', download=True, train=True, transform=transform)
valset = datasets.MNIST('C:/Users/vgiorda1/Python/ELM-with-Iterative-Optimizations/.data', download=True, train=False, transform=transform)
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


model = randomNet(784, 500, 10, torch.nn.functional.sigmoid)
elm = classifierELM(model, images.float(), labels.float(), test_images, test_labels)
init = elm.fit('SOR')
elm.classify()
