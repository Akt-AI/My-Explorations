import torch
import math

def estimate_memory(input_shape, model, batch_size, precision):
    input_size = input_shape[1] * input_shape[2] * input_shape[3]
    
    # Get the model parameters
    params = sum(p.numel() for p in model.parameters())
    
    # Estimate the size of a single parameter
    param_size = 4 if precision == 'float32' else 8
    
    # Calculate the total memory required
    total_memory = batch_size * input_size * precision + params * param_size
    
    # Convert total memory to MB
    total_memory_MB = total_memory / 1024 / 1024
    
    # Convert total memory to GB
    total_memory_GB = total_memory / 1024 / 1024 / 1024
    
    return total_memory_MB, total_memory_GB

input_shape = (1, 3, 512, 512)
model = torch.nn.Linear(3 * 512 * 512, 100)
batch_size = 64
precision = 4

total_memory_MB, total_memory_GB = estimate_memory(input_shape, model, batch_size, precision)

print(f'Total memory required for model inference with batch size {batch_size}: {total_memory_MB:.2f} MB or {total_memory_GB:.2f} GB')

