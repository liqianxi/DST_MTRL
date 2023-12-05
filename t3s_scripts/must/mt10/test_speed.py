import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define your modules
class Module1(nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 64)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Module2(nn.Module):
    def __init__(self):
        super(Module2, self).__init__()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc3(x)
        return x

# Create instances of your modules
module1 = Module1()
module2 = Module2()
#print(module1.fc1.weight)

# Some input data
input_data = torch.randn(1, 2, requires_grad=True)

# Forward pass through module1
output_module1 = module1(input_data)

# Detach the output of module1 to stop gradients from flowing through it
detached_output_module1 = output_module1.detach()

# Forward pass through module2 with the detached output
output_module2 = module2(detached_output_module1)

# Compute your loss
loss = torch.nn.functional.mse_loss(output_module2, torch.ones_like(output_module2))

# Backward pass and update only the parameters of module2
optimizer = optim.SGD([i for i in module2.parameters()]+[i for i in module1.parameters()], lr=0.01)
optimizer.zero_grad()
print("forward done")
time0 = time.time()
loss.backward()
optimizer.step()
time1 = time.time()
print("time",time1-time0)
print(len(list(module1.parameters())))
# print(module1.fc1.weight.grad)
#