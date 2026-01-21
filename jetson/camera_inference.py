import tensorrt as trt
import numpy as np
import cv2
import json

ENGINE_PATH = "/home/rushil-mohan/distracted-driver-inference/models/driveraction_fp16.engine"
CLASS_JSON = "/home/rushil-mohan/distracted-driver-inference/models/driver_class_map.json"
IMG_SIZE = 224

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

with open(CLASS_JSON) as f:
    classes = json.load(f)

n_classes = len(classes)

cap = cv2.VideoCapture(
    "nvarguscamerasrc sensor-id=0 ! nvvidconv ! "
    "video/x-raw, format=BGR ! appsink",
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

outputs = np.zeros((1, n_classes), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    inp = preprocess(frame)
    bindings = [inp.ctypes.data, outputs.ctypes.data]
    context.execute_v2(bindings)

    pred = int(np.argmax(outputs))
    label = classes[str(pred)]

    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Driver Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
