net.qconfig = torch.quantization.get_default_config('fbgemm')
# insert observers
torch.quantization.prepare(net, inplace=True)
# Calibrate the model and collect statistics
# convert to quantized version
torch.quantization.convert(net, inplace=True)

