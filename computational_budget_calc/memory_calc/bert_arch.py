import torch
import torchviz

# Define the model architecture
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

# Visualize the model architecture and save as an SVG file
torch.onnx.export(model, torch.zeros(1, 128).to('cpu'), "bert.onnx")
graph = torchviz.make_dot(torch.onnx.load("bert.onnx").graph)
graph.render("bert")

