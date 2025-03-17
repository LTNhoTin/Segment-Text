import onnx

model = onnx.load("models/sat-12l-sm/1/model.onnx")
print([inp.name for inp in model.graph.input])  # In ra danh sách các input
