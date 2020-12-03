import torch
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# download mnist
dataset = MNIST(root='data', download=True)
print("size of training data", len(dataset)) #test is separate in MNIST

test_dataset = MNIST(root='data/', train=False)
#print(dataset[0])

#%matplotlib inline

#plot first data set
image, label = dataset[0]
#plt.imshow(image,cmap='gray')
#plt.show()
#print('Label:', label)

import torchvision.transforms as transforms
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label) # the image is now a 1x28x28 tensor, dim1 keeps track of color channels so as its black and qhite it must be 1

#print(img_tensor[:, 10:15,10:15])
#print(torch.max(img_tensor), torch.min(img_tensor))
#plot the image by passing in the 28x28 matrix
#plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
#plt.show()

# training and Validation data sets
# split data into 3 parts: training, validation and test
# in MNIST there is already test data
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000]) #random is important obviously

# batching datasets
from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


#modelling, we are going to use logistic regression which is close to linear regression
import torch.nn as nn
input_size = 28*28
num_classes = 10

# the logistic regression model
model = nn.Linear(input_size, num_classes)

# if we look at our weights and bias'
print("log model wights shape", model.weight.shape)
print("log model bias shape", model.bias.shape)

# we have images of different size to the vectors and thus we use .reshape
# to use this in our model we need to extend the nn.Module class from pytorch

class MnistModel(nn.Module):
    def __init__(self): # initialising weights and biases using nn.Linear
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb): # this is when we pass a batch of inputs through the model
        xb = xb.reshape(-1, 784) # we flatten the tensor then pass into self.linear
        out = self.linear(xb)
        return out

model = MnistModel()

#print("list of our parameters after reshaping", list(model.parameters()))

for images, labels in train_loader:
    outputs = model(images)
    break

print('outputs.shape: ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
