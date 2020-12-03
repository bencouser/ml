import torch
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# download mnist
dataset = MNIST(root='data', download=True)
print(len(dataset))

test_dataset = MNIST(root='data/', train=False)
print(dataset[0])

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

print(img_tensor[:, 10:15,10:15])
print(torch.max(img_tensor), torch.min(img_tensor))
#plot the image by passing in the 28x28 matrix
#plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
#plt.show()

# training and Validation data sets
# split data into 3 parts: training, validation and test
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000]) #random is important obviously

# batching datasets
from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

