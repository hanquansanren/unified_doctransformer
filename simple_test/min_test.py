import torch
import torch.nn.functional as F
a = torch.tensor([0,0,0,0,0,0])
b = torch.tensor([1,2,3,0,0,0])

# print(a,b)
# # c = filter(lambda x: x > 0, b)
# # print(c)
# print(b.nonzero())
# print(b[b.nonzero()])
# print(torch.min(filter(lambda x: x > 0, b)))

a = torch.tensor([1,1,1,0,0,0]).float()
b = torch.tensor([1,0,0,0,0,0]).float()
# input = torch.ones(3, requires_grad=True)
# target = torch.ones(3)
loss = F.binary_cross_entropy_with_logits(a, b)

loss.backward()