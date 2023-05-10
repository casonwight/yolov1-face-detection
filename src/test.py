import torch

a = torch.rand(10, 10)
print(a)

b = torch.tensor([[1, 2], [3, 4], [1, 2], [5, 6]])
print(b)

unique_b, inverse_index = torch.unique(b, dim=0, return_inverse=True)
unique_b[inverse_index, :] = b
print(unique_b)

print(a[unique_b[:, 0], unique_b[:, 1]])