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

input_names = [engine.get_tensor_name(i) for i in range(3)]
output_name = engine.get_tensor_name(3)

for name in input_names:
    context.set_input_shape(name, (1,3,IMG_SIZE,IMG_SIZE))

output_shape = context.get_tensor_shape(output_name)

full_buf = np.empty((1,3,IMG_SIZE,IMG_SIZE), dtype=np.float32)
face_buf = np.empty((1,3,IMG_SIZE,IMG_SIZE), dtype=np.float32)
hand_buf = np.empty((1,3,IMG_SIZE,IMG_SIZE), dtype=np.float32)
out_buf = np.empty(output_shape, dtype=np.float32)

context.set_tensor_address(input_names[0], full_buf.ctypes.data)
context.set_tensor_address(input_names[1], face_buf.ctypes.data)
context.set_tensor_address(input_names[2], hand_buf.ctypes.data)
context.set_tensor_address(output_name, out_buf.ctypes.data)

cap = cv2.VideoCapture(
    "nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W = frame.shape[:2]

    full_buf[0] = preprocess(frame)
    face_buf[0] = preprocess(frame)
    hand_buf[0] = preprocess(frame[H//2:H, W//2:W])

    context.execute_async_v3(0)

    pred = int(np.argmax(out_buf))
    label = classes[str(pred)]

    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Driver Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
