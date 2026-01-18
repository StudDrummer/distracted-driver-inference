import cv2
import json
import numpy as np
import torch
from trt_inference import TRTInference


with open("../models/driverclass.json", "r") as f:
    CLASS_NAMES = json.load(f)

CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}

#load tensor engine

engine = TRTInference("../models/driver_action_fp16.engine")


#preproccessing

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BBGR2RGB)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

    img = (img - mean) / std
    img = img.transpose(2, 0, 1) #CHW
    return img

#video source

cap = cv2.VideoCapture(0) #0 - webcam

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")


#main loop

while True: 
    ret, frame = cap.read()
    if not ret: 
        break

    H, W, _ = frame.shape

    full = frame
    face = frame #replace with facedetect (for now a placeholder)
    hand = frame[H//2:H, W//2:W]


    x = np.stack([
        preprocess(full),
        preprocess(face),
        preprocess(hand)
    ], axis=0)

    x = np.expand_dims(x, axis=0) # (1, 3, 3, 224, 224)

    logits = engine.infer(x)
    pred = int(np.argmax(logits, axis=1)[0])
    label = CLASS_NAMES[pred]


    cv2.putText(
        frame,
        label, 
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0), 
        2
    )


    cv2.imshow("Driver Action Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()