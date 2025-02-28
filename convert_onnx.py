from wtpsplit import SaT
import timeit
import torch
texts = ["This is a sentence. This is another sentence."] * 1000

device ="cpu"  #--Need a same divice as model
# PyTorch CPU
model_pytorch = SaT("sat-3l-sm")
model_pytorch.half().to(device)

pytorch_time = timeit.timeit(lambda: list(model_pytorch.split(texts)), number=10)
print(f"PyTorch time: {pytorch_time/10*1000:.1f} ms per loop")

# onnxruntime CPU
providers = ["CPUExecutionProvider"] 
model_ort = SaT("sat-3l-sm", ort_providers=providers)
ort_time = timeit.timeit(lambda: list(model_ort.split(texts)), number=10)
print(f"ONNX Runtime time: {ort_time/10*1000:.1f} ms per loop")