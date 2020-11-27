import torch
import numpy as np

# input data = temp, rainfall, humidity
inputs = np.array([[73, 67, 43],
                    [91, 88, 64],
                    [87, 134, 58],
                    [102, 43, 37],
                    [69, 96, 70]], dtype='float32')

# target = apples, oranges
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# converting
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# linear regression from scratch
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
#print(w, b)

#modelling
def model(x):
    return x @ w.t() + b # x cross w(transposed) + b

preds = model(inputs)

print("predictions: ", preds, "\ntarget: ", targets)


#loss function MSE
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

loss = mse(preds, targets)
print(loss)

#compute gradients
loss.backward()

#grads of weights
print(w)
print(w.grad)

# we loss is w quadratic function of our weights and bias, we need to find the minimum of it and that is the appropriate weights and bias'
# usual calculus methods
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)

# adjusting weights and bias with gradient decent 
preds = model(inputs)
loss = mse(preds, targets)
loss.backward()

# subract by a small amount proportional to the grad
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)
print(loss)

# we have an increase in loss, but now we can automate over lots of steps
# 1e-5 is known as a hyper parameter
for i in range(1000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5 
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

# now lets verify by calculating the loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


# Linear reg with pytorch built-ins
# nn = neural networks
import torch.nn as nn

#inputs
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# dataset and dataloaders
from torch.utils.data import TensorDataset

#define dataset
train_ds = TensorDataset(inputs,targets)

#dataloader can split data into batches of a predefined size while training
# it can also shuffle and random sample
from torch.utils.data import DataLoader

#define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# nn.linear
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

# we can list parameters with
list(model.parameters())
print(list(model.parameters()))

#now we can generate predictions like before
preds = model(inputs)
print("nn preds", preds)

# nn lose function
import torch.nn.functional as F

loss_fn = F.mse_loss
losdd = loss_fn(model(inputs), targets)
print(loss)

#optimisation
opt = torch.optim.SGD(model.parameters(), lr=1e-5) #SGD = stochastic gradient descent

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(100, model, loss_fn, opt, train_dl)

#generate preds
preds = model(inputs)
print("linear neural network preds: ", preds)
print("targets: ", targets)

model(torch.tensor([[75,63,44.]]))
print(model(torch.tensor([[75, 63, 44.]])))
