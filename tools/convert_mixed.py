import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name):
    input = torch.randn(1, 3, 128, 128)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    with torch.cuda.amp.autocast():
    	torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    checkpoint_path = ""
    net = PoseEstimationWithMobileNet()
    net.eval()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    load_state(net, checkpoint)
    
    output_name = 'model.onnx'
    convert_to_onnx(net, output_name)
