import torch.nn.utils.prune as prune
from models.with_mobilenet import PoseEstimationWithMobileNet_RGB, PoseEstimationWithMobileNet2 
import torch
from modules.load_state import load_state
from torch import nn
from modules.conv import conv


def tf_learn():
    model = PoseEstimationWithMobileNet_RGB()
    modules = list(model.children())
    #modules[0]: Base, modules[1]: Cpm, modules[2]:InitialStage, modules[3]:RefinementStage
    for block in range(len(modules)):
        if block in range(0, 2):
            for param in modules[block].parameters():
                param.requires_grad = False
                   
        if block==3:
            layers = []
            for block_layers in modules[block][0].children():
                for layer in  block_layers.children():
                    layers.append(layer)
                    
            for layer in layers[0:2]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    return model
    
input = torch.zeros(1, 3, 224,224)
model = tf_learn()
out = model(input)
from torchviz import make_dot

print(out)
graph = make_dot(out[0]).render() 


                   
