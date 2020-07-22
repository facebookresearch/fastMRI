import torch


dtype = torch.float
device = torch.device("cuda:0")

for i in range(100000000000):
    N, D_in, H, D_out = 64, 1000, 100, 10
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device, dtype=dtype)

