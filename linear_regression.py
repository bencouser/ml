import torch
import numpy as np

# input == (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                    [91, 88, 64],
                    [87, 134, 58],
                    [102, 43, 37],
                    [69,96,70]], dtype='float32')

# targets == (apples, organges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')


# linear regression model from scratch
#yield_apple = w11 * temp + w12 * rainfall + w13 * humidity + b1
#yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2

w = torch.randn(2, 3, required_grad=True)
b = torch.randn(2, required_grad=True)

print(b, w)
