import torch 
import os
import torchvision


def network_to_half(model):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    def bn_to_float(module):
        """
        BatchNorm layers need parameters in single precision. Find all layers and convert
        them back to float.
        """
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            bn_to_float(child)
        return module
    return bn_to_float(model.half())


model = torchvision.models.resnet18(pretrained=True).eval().half()
print(model)

# Convert model to have
"""
model = network_to_half(model)

dummy_input = torch.randn(1, 3, 224, 224).half()

torch.onnx.export(model, dummy_input, "test.onnx", verbose=False)
os.system('onnx2trt -w 12000000000 -b 1 -d 16 test.onnx -o test.engine')"""
