import torch
import torch.nn as nn

# Define a single linear layer
class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

# Create a dummy input tensor with shape (batch_size, input_size)
input_size = 1024
batch_size = 128
dummy_input = torch.randn(batch_size, input_size)

# Create the linear layer and move it to CPU
model = LinearLayer(input_size, 128)
model.to('cpu')

# Use the `torch.cuda.memory_allocated` method to get the total GPU memory usage in bytes
memory_allocated = torch.cuda.memory_allocated()

# Convert memory_allocated from bytes to MB
memory_allocated_MB = memory_allocated / 1024**2

# Convert memory_allocated from bytes to GB
memory_allocated_GB = memory_allocated / 1024**3

print(f'Total memory used by the model: {memory_allocated_MB:.2f} MB or {memory_allocated_GB:.2f} GB')

