import argparse
import torch

from src.with_mobilenet import PoseEstimationWithMobileNet
from src.load_state import load_state
from pathlib import Path
import os


filepath = 'model.onnx'
model = SimpleModel()
input_sample = torch.randn((1, 64))
model.to_onnx(filepath, input_sample, export_params=True)

ort_session = onnxruntime.InferenceSession(filepath)
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: np.random.randn(1, 64).astype(np.float16)}
ort_outs = ort_session.run(None, ort_inputs)

model = SimpleModel()
script = model.to_torchscript()

# save for use in production environment
torch.jit.save(script, "model.pt")
