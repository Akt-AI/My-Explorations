ffmpeg -i cam1_fall1.mp4 -f image2 -vf "select='eq(pict_type,PICT_TYPE_I)'" -vsync vfr out/frame%03d.png
