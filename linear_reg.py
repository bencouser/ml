# something went wrong, restart linear regression section.....

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

# now lets verify

preds = model(inputs)
loss = mse(preds, targets)
print(loss)

