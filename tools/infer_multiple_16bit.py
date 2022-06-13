import argparse
import math
import time
import numpy as np
import onnxruntime as rt
import cv2
from src.process_keypoints import extract_keypoints, group_keypoints
from src.process_pose import ProcessPose, pose_tracking
from pathlib import Path
import os


def convert_fps_to_frames_per_millisecond(frame):
    fps = frame #frames per second
    time_one_frame_persecond = 1/fps
    time_one_frame_per_milli_second = time_one_frame_persecond//0.001
    return time_one_frame_per_milli_second


class ReadImages(object):
    """This class contains Methods of Image Reading.

    Parameters
    ----------
    Inputs: file_names
    
    Returns
    -------
        __next__ method returns img: Image
    """
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class ReadVideo(object):
    """This class contains Methods of opening and 
    reading image frames from video and livestream.

    Parameters
    ----------
    Inputs: file_name
    
    Returns
    -------
        __next__ method returns img: Image
    """
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def normalize_image(img, img_mean, img_scale):
    """This function normalizes input image.

    Parameters
    ----------
    Inputs: img, img_mean, img_scale
    img: 
        Image
    img_mean: 
        Image mean
    img_scale: 
        Image scale
    
    Returns
    -------
        This function returns Normalized image: img.
        
    """
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def get_pad_width(img, stride, pad_value, min_dims):
    """This function calculates padding width and returns padded image.

    Parameters
    ----------
    Inputs: img, stride, pad_value, min_dims
    img:
        Image
    stride:
        Stride
    pad_value:
        Padding Value
    min_dims:
        Minimum dimensions

    Returns
    -------
        This Function Returns Padded Image and padding list: padded_img, pad
        
    """
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def predict(sess, label_names, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    """This Function Preprocess Image Input and Make Inferance from Model.

    Parameters
    ----------
    Inputs: sess, label_names, img, net_input_height_size, stride, upsample_ratio,
            pad_value, img_mean, img_scale
    sess:
        Onnx Session variable 
    label_names:
        Input label names
    img:
        Image 
    net_input_height_size:
        Input image height size 
    stride:
        Stride value
    upsample_ratio:
        Upsampling ratio
    pad_value:
        Padding value: Default value is (0, 0, 0).
    img_mean:
        Image mean: Default value is (128, 128, 128). 
    img_scale: 
        img_scale: Default value is (1/256).
      
    Returns
    -------
        This function returns model predictions: heatmaps, pafs, scale, pad
    """
    height, width, _ = img.shape
    scale = net_input_height_size / height    
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize_image(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    _, pad = get_pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = np.expand_dims(scaled_img, axis=0)
    tensor_img = np.moveaxis(tensor_img, -1, 2)
    tensor_img = np.moveaxis(tensor_img, -2, 1)
    stages_output = sess.run(label_names, {input_name: tensor_img.astype(np.float16)})
    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, 
                               fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, 
                                                             interpolation=cv2.INTER_CUBIC)
    return heatmaps, pafs, scale, pad
    
    
def show_results(sess, label_names, image_provider, height_size, track, smooth, output_path):
    """This function exracts Keypoints from Model predictions and shows results.

    Parameters
    ----------
    Inputs: sess, label_names, image_provider, height_size, track, smooth
    sess:
        Onnx Session variable 
    label_names: 
        Input label names
    image_provider: 
        Read Images
    height_size:
        Input image height size 
    track:
        Defaults value is 1.
    smooth:
        Defaults value is 1.
    
    Returns
    -------
        This functions Returns None
    """
    stride = 8
    upsample_ratio = 4
    num_keypoints = ProcessPose.num_kpts
    previous_poses = []
    delay = 33
    count = 0
    start = time.time()
    writer = cv2.VideoWriter_fourcc(*'DIVX')
    frame_list = []
    for img in image_provider:
        count = count+1
        #img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE) 
        img = cv2.resize(img, (512, 512))
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = predict(sess, label_names, 
                                      img, height_size, stride, upsample_ratio)
        #print('Average FPS:', count / (time.time() - start))
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], 
                                            all_keypoints_by_type, total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = ProcessPose(pose_keypoints, pose_entries[n][18])
            
            #print(pose_keypoints)
            current_poses.append(pose)
    
        #print('Average FPS:', count / (time.time() - start))
        
        if track:
            pose_tracking(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img, [0, 255, 0])
        
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 0, 255))
            if track:
                pass
                #cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            #cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        
        cv2.imshow('Demo', img)
        cv2.waitKey(1)
        #frame_list.append(img)
    
        cv2.imwrite(output_path + os.path.sep + 'frame' +str(count)+".jpg", img)        
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
            
    '''for i in range(len(frame_list)):
        img = frame_list[i]
        height, width, _= img.shape
        size = (width,height)
        out = cv2.VideoWriter(output_path+'.mp4', writer, 30, size)
        out.write(img)
    out.release()
    frame_list.clear()'''
    
    fps = count / (time.time() - start)
    #print('Average FPS:', count / (time.time() - start))
    print('Average FPS:', fps) 
        
    perfromance = convert_fps_to_frames_per_millisecond(fps)
    print("Perfromance:", perfromance, "Milli Second")
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Pose Model''')
    parser.add_argument('--height-size', type=int, default=256, 
                                  help='Model input layer height size')
    #parser.add_argument('--video', type=str, default='', help='Location of video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='Input image path')  
    parser.add_argument('--track', type=int, default=1, help='Track pose id')
    parser.add_argument('--smooth', type=int, default=1, help='Smooth out pose keypoints')
    parser.add_argument('--onnx-model-path', type=str, default='onnx_dir', help='Onnx model Path')
    args = parser.parse_args()

    '''if args.video == '' and args.images == '':
        raise ValueError('Use --video or --image options and provide video or image file location.')'''

    models_path = args.onnx_model_path
    models_path = Path(models_path)
    video_input_path = Path('input')
    for model_name in models_path.iterdir():
        model_name = str(model_name)
        print(model_name)
        sess = rt.InferenceSession(model_name)
        input_name = sess.get_inputs()[0].name
        label_names = sess.get_outputs()
        label_names = [label_name.name for label_name in label_names]
        frame_provider = ReadImages(args.images)
        for video_filename in video_input_path.iterdir():
            video_filename = str(video_filename)
            
            if video_filename != '':
                frame_provider = ReadVideo(video_filename)
            else:
                args.track = 0
            '''if args.video != '':
                frame_provider = ReadVideo(args.video)
            else:
                args.track = 0'''

            start_time = time.time()
            output_path = 'outputs/'+model_name.split('/')[-1].split('.')[0]
            video_ouput_path = output_path+os.path.sep+video_filename.split('/')[-1].split('.')[0]
            #print(video_ouput_path)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            if not os.path.exists(video_ouput_path):
                os.mkdir(video_ouput_path)
                
            #print(output_path)
            show_results(sess, label_names, frame_provider, args.height_size, args.track, args.smooth, video_ouput_path)
            print("########### Running Inferance ############")
            print("Inferance Completed.....")
