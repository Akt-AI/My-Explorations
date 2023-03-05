import psutil
import torch
import os
import time

# Get the current process information
process = psutil.Process(pid=os.getpid())

# Calculate the current CPU memory usage before creating the model and inputs
start_memory = process.memory_info().rss

# Create the model and inputs

# Define your model
model = torch.nn.Linear(100, 200)
inputs = torch.rand(100)
# Switch to evaluation mode and prevent gradient computation
# Switch to evaluation mode and prevent gradient computation
model.eval()
with torch.no_grad():
    # Start measuring the time
    start_time = time.time()

    # Perform inference on the inputs
    output = model(inputs)

    # Stop measuring the time
    end_time = time.time()

# Calculate the current CPU memory usage after inference
end_memory = process.memory_info().rss

# Calculate the memory usage
memory_usage = end_memory - start_memory

# Convert the memory usage to MB
memory_usage_mb = memory_usage / (1024 * 1024)
print("Memory usage:", memory_usage_mb, "MB")

# Calculate the time usage
time_usage = end_time - start_time
print("Time usage:", time_usage, "seconds")

