import argparse
import torch
from src.with_mobilenet import PoseEstimationWithMobileNet
from src.load_state import load_state
from pathlib import Path
import os


def convert_to_onnx(net, output_name):
    input = torch.randn(1, 3, 256, 256)
    #input = torch.randn(1, 3, 128, 128)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default = 'pruned_quantized', required=False, help='path to the checkpoint')
    parser.add_argument('--output-folder', type=str, default='onnx_dir',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    checkpoints_path = Path('pruned_quantized')
    for checkpoint_path in checkpoints_path.iterdir():
        checkpoint_path = str(checkpoint_path)
        #print(checkpoint_path)
        net = PoseEstimationWithMobileNet()
        net.eval()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        load_state(net, checkpoint)
        
        
        if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)
            
        output_name = str(args.output_folder) + os.path.sep + checkpoint_path.split('/')[-1].split('.')[0]+'.onnx'
        #output_name = 'onnx/'+ checkpoint_path.split('/')[-1].split('.')[0]+'.onnx'
        convert_to_onnx(net, output_name)
