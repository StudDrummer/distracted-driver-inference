import tensorrt as trt
import numpy as np
import cv2
import json

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Paths
ENGINE_PATH = "models/driveraction_fp16.engine"
CLASS_JSON = "driverclass.json"
TEST_IMAGE = "test.jpg"

# Load engine
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# Load class labels
with open(CLASS_JSON) as f:
    classes = json.load(f)

# Load and preprocess image
img = cv2.imread(TEST_IMAGE)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC → CHW
img = np.expand_dims(img, axis=0)

# Allocate buffers
inputs = np.ascontiguousarray(img)
outputs = np.zeros((1, len(classes)), dtype=np.float32)
bindings = [inputs.ctypes.data, outputs.ctypes.data]

# Run inference
context.execute_v2(bindings)
pred = np.argmax(outputs)
print("Prediction:", classes[str(pred)])
