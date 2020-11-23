import torch

#vector
t2 = torch.tensor([1.,2,3,4])

print(t2)


#matrix
t3 = torch.tensor([[5., 6], [7, 8], [9, 10]])

print(t3)

#3d array

t4 = torch.tensor([
    [[11, 12, 13],
    [13, 14, 15]],
    [[15, 16, 17],
    [17, 18, 19.]]])

print(t4)

print(t2.shape, t3.shape, t4.shape)

# tensor operations and gradients or derivatives

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b

print(y, y.size)
print(y.backward())

print('dy/dw', w.grad)

#numpy integrated with torch

import numpy as np

x = np.array([[1, 2], [3, 4]])

# array into tensor
y = torch.from_numpy(x)

print(x.dtype, y.dtype)

z = y.numpy()
print(z)



