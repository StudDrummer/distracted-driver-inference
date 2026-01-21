import cv2
import numpy as np
import json
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

MODEL_PATH = "../models/driveraction_fp16.engine"
CLASS_MAP_PATH = "../models/driver_class_map.json"

# Load class map
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)

# Load TensorRT engine
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Prepare context
engine = load_engine(MODEL_PATH)
context = engine.create_execution_context()

inputs, outputs, bindings, stream = [], [], [], None
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = np.empty(size, dtype=dtype)
    inputs.append(host_mem) if engine.binding_is_input(binding) else outputs.append(host_mem)
    bindings.append(host_mem.ctypes.data)

# Load and preprocess an image
img = cv2.imread("test.jpg")  # Replace with your test image path
img = cv2.resize(img, (224, 224))  # Resize as per your training
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  
img = np.expand_dims(img, axis=0)  
inputs[0][:] = img.ravel()

# Run inference
context.execute_v2(bindings)

output = outputs[0].reshape(1, -1)
pred_class = np.argmax(output)
print("Predicted class:", class_map[str(pred_class)])
