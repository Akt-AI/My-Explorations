import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16


def onnx_32_to_16_converter(*args):
	"""The float16converter tool converts the float32 tensor type to the float16 tensor type 
    for an input ONNX model. This is especially useful for quantization support. """
	
	# Load your model
	onnx_model = onnxmltools.utils.load_model(input_onnx_model)

	# Convert tensor float type from your input ONNX model to tensor float16
	onnx_model = convert_float_to_float16(onnx_model)

	# Save as protobuf
	onnxmltools.utils.save_model(onnx_model, output_onnx_model)
	
if __name__=="__main__":
	input_onnx_model = '32_model.onnx'
	output_onnx_model = 'model_f16.onnx'
	onnx_32_to_16_converter(input_onnx_model, output_onnx_model)
	print("Conversion to float16 model completed Successfully.")
