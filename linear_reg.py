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

def model(x):
    return x @ w.t() + b # x cross w(transposed) + b

preds = model(inputs)

print("predictions: ", preds, "\ntarget: ", targets)
