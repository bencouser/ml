import torch
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# download mnist
dataset = MNIST(root='data', download=True)
print("size of training data", len(dataset)) #test is separate in MNIST

test_dataset = MNIST(root='data/', train=False)

import torchvision.transforms as transforms
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

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
        self.linear = nn.Linear(input_size, num_classes) # making nn.linear part of this class

    def forward(self, xb): # this is when we pass a batch of inputs through the model
        xb = xb.reshape(-1, 784) # we flatten the tensor then pass into self.linear
        out = self.linear(xb) # -1 means the it can now be any batch size 
        return out

model = MnistModel()

for images, labels in train_loader:
    outputs = model(images)
    break

print('outputs.shape: ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)

#our output we want to be 10 probabilities of it being 0,9 these arnt probabilites and thus
# we must change them to between 0,1 and all should add to 1
# we will use the softmax function
import torch.nn.functional as F

# applying sm
probs = F.softmax(outputs, dim=1)
print("sample probabilies:\n", probs[:2].data, "Sum: ", torch.sum(probs[0]).item())

#find predictions by using max prob in probs output
max_probs, preds = torch.max(probs, dim=1)
#print(preds)
#print(labels) we find they are not the same as we have not done any regression

#evaluate matric and loss funtions



