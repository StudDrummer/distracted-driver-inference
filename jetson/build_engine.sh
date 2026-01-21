#!/bin/bash
echo "Building TensorRT engine from ONNX..."

# Path to ONNX model
ONNX_PATH="../models/driver_action.onnx"   # make sure underscore matches

# Output engine path
ENGINE_PATH="../models/driveraction_fp16.engine"

# Build engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --fp16 \
  --verbose