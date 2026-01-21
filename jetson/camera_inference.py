import tensorrt as trt
import numpy as np
import cv2
import json
import time

ENGINE_PATH = "/home/rushil-mohan/distracted-driver-inference/models/driveraction_fp16.engine"
CLASS_JSON = "/home/rushil-mohan/distracted-driver-inference/models/driver_class_map.json"
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
CONF_THRESHOLD = 0.5
IMG_SIZE = 224
SKIP_FRAMES = 2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
n_classes = len(json.load(open(CLASS_JSON)))

with open(CLASS_JSON) as f:
    classes = json.load(f)

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGR ! appsink"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
assert cap.isOpened(), "Camera not opened!"

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    return np.expand_dims(img, axis=0).astype(np.float32)

last_face = None
frame_count = 0
def detect_face(frame):
    global last_face, frame_count
    frame_count += 1
    if frame_count % SKIP_FRAMES != 0 and last_face is not None:
        return last_face
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    if detections.shape[2] > 0:
        conf = detections[0,0,0,2]
        if conf > CONF_THRESHOLD:
            box = detections[0,0,0,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            dx = int(0.25*(x2-x1))
            dy = int(0.25*(y2-y1))
            x1,y1 = max(0,x1-dx), max(0,y1-dy)
            x2,y2 = min(w,x2+dx), min(h,y2+dy)
            last_face = frame[y1:y2, x1:x2]
            return last_face
    last_face = frame.copy()
    return last_face

def get_hand_roi(frame):
    H,W = frame.shape[:2]
    return frame[H//2:H, W//2:W]

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    full = frame
    face = detect_face(frame)
    hand = get_hand_roi(frame)
    full_t = preprocess(full)
    face_t = preprocess(face)
    hand_t = preprocess(hand)
    combined_input = np.concatenate([full_t, face_t, hand_t], axis=1)
    outputs = np.zeros((1, n_classes), dtype=np.float32)
    bindings = [combined_input.ctypes.data, outputs.ctypes.data]
    context.execute_v2(bindings)
    pred = np.argmax(outputs)
    label = classes[str(pred)]
    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Driver Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
