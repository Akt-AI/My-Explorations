import onnxruntime as ort
from imutils.video import FPS


options = ort.SessionOptions()
options.enable_profiling = True
ort_session = ort.InferenceSession('model_16.onnx', options)
outputs = ort_session.run(None, {'input': images[0].cpu().numpy()})
prof_file = ort_session.end_profiling()


fps = FPS().start()
for i in range(100):
    images = list(image.to('cuda:0') for image in x)
    with autocast():
        pred = model(images)
    fps.update()

fps.stop()
print('Time taken: {:.2f}'.format(fps.elapsed()))
print('~ FPS : {:.2f}'.format(fps.fps()))


import onnxruntime as ort
ort_session = ort.InferenceSession('model_16.onnx')
fps = FPS().start()

for i in range(100):
    outputs = ort_session.run(None, {'input': images[0].cpu().numpy()})
    fps.update()

fps.stop()
print('Time taken: {:.2f}'.format(fps.elapsed()))
print('~ FPS : {:.2f}'.format(fps.fps()))

