import tensorrt as trt
import numpy as np
import cv2
import json

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("driveraction_fp16.engine")
context = engine.create_execution_context()

# Load class names
with open("driverclass.json") as f:
    classes = json.load(f)

# Load and preprocess image
img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Allocate buffers
input_shape = img.shape
output_shape = (1, len(classes))

inputs = np.ascontiguousarray(img)
outputs = np.zeros(output_shape, dtype=np.float32)

bindings = [inputs.ctypes.data, outputs.ctypes.data]

context.execute_v2(bindings)

pred = np.argmax(outputs)
print("Prediction:", classes[str(pred)])
