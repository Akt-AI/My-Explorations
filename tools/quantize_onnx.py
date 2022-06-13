import onnx
from onnx import *
from quantize import quantize, QuantizationMode
from quantize import *
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType



"""
# Load the onnx model     
model = onnx.load('checkpoint_iter_340000_epochId_59pruned.onnx')

# Quantize
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
 
# Save the quantized model
onnx.save(quantized_model, 'qunatized_pose.onnx')
"""
model_fp32 = 'checkpoint_iter_340000_epochId_59pruned.onnx'
model_quant = 'model_quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)


