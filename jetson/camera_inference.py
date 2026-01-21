import tensorrt as trt
import numpy as np
import cv2
import json

ENGINE_PATH = "/home/rushil-mohan/distracted-driver-inference/models/driveraction_fp16.engine"
CLASS_JSON = "/home/rushil-mohan/distracted-driver-inference/models/driver_class_map.json"

# Face detector (OpenCV DNN, very lightweight)
FACE_PROTO = "deploy.prototxt"   # download from opencv github
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"  # OpenCV SSD face
CONF_THRESHOLD = 0.5
IMG_SIZE = 224

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()


with open(CLASS_JSON) as f:
    classes = json.load(f)

# Load face detector
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


def preprocess(img):
    """Resize, normalize, CHW, add batch dim"""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    return np.expand_dims(img, axis=0)

def detect_face(frame):
    """Detect first face, return ROI coordinates"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    if detections.shape[2] > 0:
        conf = detections[0,0,0,2]
        if conf > CONF_THRESHOLD:
            box = detections[0,0,0,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            # expand box slightly
            dx = int(0.25*(x2-x1))
            dy = int(0.25*(y2-y1))
            x1,y1 = max(0,x1-dx), max(0,y1-dy)
            x2,y2 = min(w,x2+dx), min(h,y2+dy)
            return frame[y1:y2, x1:x2]
    return frame.copy()  # fallback
def get_hand_roi(frame):
    """Bottom-right quarter crop (as in original preprocessing)"""
    H,W = frame.shape[:2]
    return frame[H//2:H, W//2:W]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare ROIs
    full = frame.copy()
    face = detect_face(frame)
    hand = get_hand_roi(frame)

    # Preprocess
    full_t = preprocess(full)
    face_t = preprocess(face)
    hand_t = preprocess(hand)

    # Allocate buffers
    inputs = [np.ascontiguousarray(full_t), 
              np.ascontiguousarray(face_t), 
              np.ascontiguousarray(hand_t)]
    combined_input = np.concatenate(inputs, axis=1)
    outputs = np.zeros((1,len(classes)), dtype=np.float32)
    bindings = [combined_input.ctypes.data, outputs.ctypes.data]

    # Run inference
    context.execute_v2(bindings)
    pred = np.argmax(outputs)
    label = classes[str(pred)]

    # Overlay prediction
    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Driver Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()