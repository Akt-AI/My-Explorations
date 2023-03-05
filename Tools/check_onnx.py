import onnx
import onnxruntime
import numpy as np
import torch

onnx_model = onnx.load("optimized_alphapose.onnx")
results = onnx.checker.check_model(onnx_model)
print(results)

ort_session = onnxruntime.InferenceSession("alpha_pose.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.randn(1, 3, 256, 192)
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")


passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
from onnx import optimizer
optimized_model = optimizer.optimize(onnx_model, passes)

onnxfile = "optimized_alphapose.onnx"
onnx.save(optimized_model, onnxfile)
