#!/bin/bash
set -e

ONNX_MODEL=../models/driver_action.onnx
ENGINE_OUT=../models/driver_action_fp16.engine

echo "Building TensorRT engine from ONNX..."
echo "ONNX: $ONNX_MODEL"
echo "ENGINE: $ENGINE_OUT"

if [ ! -f "$ONNX_MODEL" ]; then
  echo "ERROR: ONNX model not found!"
  exit 1
fi

/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_MODEL \
  --saveEngine=$ENGINE_OUT \
  --fp16 \
  --workspace=4096 \
  --explicitBatch \
  --verbose

echo "TensorRT engine built successfully."


chmod +x build_engine.sh
