import os

def get_memory_size(num_params, precision):
    # Calculate memory size in bytes
    memory_size = num_params * precision / 8
    # Convert to gigabytes
    memory_size = memory_size / (1024**3)
    return memory_size

num_params = 17000000000 # 1 billion parameters
precision = 32 # using 32-bit precision

memory_size = get_memory_size(num_params, precision)
print("Total memory required by the model: {:.2f} GB".format(memory_size))


import torch

# Define your model
model = torch.nn.Linear(100, 200)

# Get the total number of parameters
num_params = sum(p.numel() for p in model.parameters())

print(num_params)
#print("Allocated GPU memory:", torch.cuda.memory_allocated())

import torch



# Transfer the model to GPU (if available)
if torch.cuda.is_available():
    model.cuda()

# Calculate the total GPU memory usage
if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated()
    print("Allocated GPU memory:", gpu_memory)
else:
    gpu_memory = 0

# Calculate the total CPU memory usage
cpu_memory = torch.cuda.max_memory_allocated()
print("Allocated CPU memory:", cpu_memory)

# Calculate the total memory usage
total_memory = gpu_memory + cpu_memory
print("Total memory usage:", total_memory)


'''
# Transfer the model to GPU (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")

# Calculate the memory usage summary
print(torch.cuda.memory_summary(device=device))'''



# Switch to evaluation mode and prevent gradient computation
model.eval()
with torch.no_grad():
    # Calculate the total GPU memory usage before inference
    start_memory = torch.cuda.memory_allocated()

    inputs = torch.rand(100)
    # Perform inference on the inputs
    output = model(inputs)

    # Calculate the total GPU memory usage after inference
    end_memory = torch.cuda.memory_allocated()

# Calculate the memory usage
memory_usage = end_memory - start_memory
print("Memory usage:", memory_usage)


model.eval()
with torch.no_grad():
    # Calculate the total CPU memory usage before inference
    start_memory = torch.cuda.max_memory_allocated()

    # Perform inference on the inputs
    output = model(inputs)

    # Calculate the total CPU memory usage after inference
    end_memory = torch.cuda.max_memory_allocated()

# Calculate the memory usage
memory_usage = end_memory - start_memory
print("Memory usage:", memory_usage)

import psutil
import torch

# Get the current process information
process = psutil.Process(pid=os.getpid())

# Calculate the current CPU memory usage
memory_info = process.memory_info().rss
print("CPU memory allocated:", memory_info)

memory_info_mb = memory_info / (1024 * 1024)
print("CPU memory allocated:", memory_info_mb, "MB")

# Convert the memory usage to GB
memory_info_gb = memory_info / (1024 * 1024 * 1024)
print("CPU memory allocated:", memory_info_gb, "GB")






