import torch
import torchvision


def alpha_pose_to_onnx():
	dummy_input = torch.randn(1, 3, 256, 192, device='cuda')
	model = pose_model
	model.eval()
	input_names = [ "preact" ]
	output_names = [ "conv_out" ]
	torch.onnx.export(model, dummy_input, 
	         "alpha_pose.onnx", 
	         verbose=True, input_names=input_names, output_names=output_names)
	         
if __name__=="__main__":
    alpha_pose_to_onnx()
