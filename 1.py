import torch

x = torch.randn(8, 4)
w = torch.randn(4, 3)
b = torch.zeros(3)

y = x @ w + b

print(x.shape)
print(w.shape)
print(y.shape)