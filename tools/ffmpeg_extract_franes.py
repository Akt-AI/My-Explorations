import os 
from pathlib import Path


input_videos = Path("inputs")
count = 0
for video in input_videos.iterdir():
    video_file = str(video)
    file_name = video_file.split("/")[-1].split('.')[0]
    output_folder = "outputs/"+ video_file.split("/")[-1].split('.')[0]
    os.mkdir(output_folder)
    count += 1 
    #command = "ffmpeg -i compq_DSCN0013.mp4 -vf fps=5 images/compq_DSCN0013_%05d.png"
    command = "ffmpeg -i " + " " + video_file + \
                   " "+ "-vf  fps=5  " + output_folder + os.path.sep + "frame%05d.png"
    
    os.system(command)


