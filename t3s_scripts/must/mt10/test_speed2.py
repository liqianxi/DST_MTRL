import torch
import timeit

# Create random tensors
tensor1 = torch.randn(10, 256, 160000).to("cuda:0")
tensor2 = torch.randn(10, 6000, 4000).to("cuda:0")

# Benchmark torch.matmul
def matmul_benchmark():
    result = torch.matmul(tensor1, tensor2)
    print("done1")

# Benchmark torch.bmm
def bmm_benchmark():
    result = torch.bmm(tensor1, tensor2)
    print("done2")

# Measure execution time for torch.matmul
matmul_time = timeit.timeit(matmul_benchmark, number=30)

# Measure execution time for torch.bmm
bmm_time = timeit.timeit(bmm_benchmark, number=30)

# Print the results
print(f"torch.matmul execution time: {matmul_time:.6f} seconds")
print(f"torch.bmm execution time: {bmm_time:.6f} seconds")
