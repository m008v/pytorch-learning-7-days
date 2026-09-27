import torch

x = torch.randn(8, 4) # ma trận 8 hàng 4 cột
w = torch.randn(4, 3) # ma trận 4 hàng 3 cột
b = torch.zeros(3)    # bias vector với 3 phần tử

y = x @ w + b 

"""@ là toán tử nhân ma trận trong PyTorch."""

print(x.shape)
print(w.shape)
print(y.shape)