import torch
import torch.nn as nn
import torch.optim as optim
import time

import timeit


a = torch.randn((8))
print(a.shape)
print(a.repeat(128,1).shape)


# # # Define your modules
# class Module1(nn.Module):
#     def __init__(self):
#         super(Module1, self).__init__()
#         self.fc1 = nn.Linear(2, 10000)
#         self.fc2 = nn.Linear(10000, 10000)
#         self.fc3 = nn.Linear(10000, 64)


#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return x

# # class Module2(nn.Module):
# #     def __init__(self):
# #         super(Module2, self).__init__()
# #         self.fc3 = nn.Linear(64, 1)

# #     def forward(self, x):
# #         x = self.fc3(x)
# #         return x

# # # Create instances of your modules

# # module2 = Module2()
# # #print(module1.fc1.weight)

# def operations(input1, net):
#     out = net(input1)
#     loss = torch.nn.functional.mse_loss(out, torch.ones_like(out))
#     return loss

# module1 = Module1()
# module2 = Module1()
# optimizer = optim.SGD([i for i in module2.parameters()]+[i for i in module1.parameters()], lr=0.01)
# input_data = torch.randn(1, 2, requires_grad=True)
# input_dat2 = torch.randn(1, 2, requires_grad=True)
# def test1():

#     # # Some input data


#     # Forward pass through module1
#     loss1 = operations(input_data, module1)
#     loss2 = operations(input_dat2, module2)
    
#     optimizer.zero_grad()

#     loss = loss1+loss2
#     loss.backward()
#     optimizer.step()

# test1_time = timeit.timeit(test1, number=3)
# del module1,module2
# del input_data,input_dat2

# module1 = Module1()
# module2 = Module1()
# optimizer1 = optim.SGD([i for i in module1.parameters()], lr=0.01)
# optimizer2 = optim.SGD([i for i in module2.parameters()], lr=0.01)
# # # Some input data
# input_data = torch.randn(1, 2, requires_grad=True)
# input_dat2 = torch.randn(1, 2, requires_grad=True)
# def test2():



#     # Forward pass through module1
#     loss1 = operations(input_data, module1)
#     loss2 = operations(input_dat2, module2)
    
#     optimizer1.zero_grad()


#     loss1.backward()
#     optimizer1.step()

#     optimizer2.zero_grad()

#     loss2.backward()
#     optimizer2.step()

# test2_time = timeit.timeit(test2, number=3)

# print(test1_time,test2_time)

# # Detach the output of module1 to stop gradients from flowing through it
# detached_output_module1 = output_module1.detach()

# # Forward pass through module2 with the detached output
# output_module2 = module2(detached_output_module1)

# # Compute your loss
# loss = torch.nn.functional.mse_loss(output_module2, torch.ones_like(output_module2))

# # Backward pass and update only the parameters of module2

# time1 = time.time()
# print("time",time1-time0)
# print(len(list(module1.parameters())))
# # print(module1.fc1.weight.grad)
# #



# def gumbel_softmax(logits, k, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
#     gumbels = (
#         -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
#     )  # ~Gumbel(0,1)
#     gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
#     y_soft = gumbels.softmax(dim)


#     if hard:
#         # Straight through.
#         #index = y_soft.max(dim, keepdim=True)[1]
#         index = torch.topk(y_soft, k, dim=-1)[1]

#         y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0).to("cuda:0")
#         ret = y_hard - y_soft.detach() + y_soft
#     else:
#         # Reparametrization trick.
#         ret = y_soft
#     return ret
    
# # Random input tensors
# a = torch.randn(10000, requires_grad=True).cuda()
# b = torch.randn(1000, requires_grad=True).cuda()
# c = torch.randn(1000, requires_grad=True).cuda()
#list1= [i for i in range(100000)]
# mask_buffer_copy = [torch.randn(100).cuda() for i in range(10)]
# def c1():

#     new_mask_list = [msk_tensor.to("cpu") for msk_tensor in mask_buffer_copy]
    

# def c2():
#     new_mask_list = [torch.as_tensor(msk_tensor,device="cpu") for msk_tensor in mask_buffer_copy]
#     #a = torch.tensor(list1,device="cuda:0")

# # def c3():
# #     a = torch.as_tensor(list1,device="cuda:0")
# # Original separate operations

# matmul_time = timeit.timeit(c1, number=3)

# # Measure execution time for torch.bmm
# bmm_time = timeit.timeit(c2, number=3)


# # time3 = timeit.timeit(c3, number=30)

# # Print the results
# print(f"torch.matmul execution time: {matmul_time:.6f} seconds")
# print(f"torch.bmm execution time: {bmm_time:.6f} seconds")
# # print(f"time3 execution time: {time3:.6f} seconds")



# a= torch.zeros((4,3))
# b= torch.ones((4,3))
# c = torch.cat([a,b])
# print(c)
# print(c.view(2,4,-1))
# print(c.view(2,4,-1).shape)
# print(c.view(2,4,-1).permute(0,2,1))
