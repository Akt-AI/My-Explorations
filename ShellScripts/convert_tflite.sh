IMAGE_SIZE=401
tflite_convert \
  --graph_def_file=/home/arun/workspace/keras_implementation_posenet/posenet_tf2/frozen_model_posenet.pb \
  --output_file=optimized_graph.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
  --input_array=Placeholder \
  --output_array=ResizeBilinear,ResizeBilinear_1,concat_256,concat_529,ResizeBilinear_4 \
  --inference_type=FLOAT \
  --input_data_type=FLOAT \
  --enable_v1_converter
