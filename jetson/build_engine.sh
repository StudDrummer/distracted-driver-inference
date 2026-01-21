#!/bin/bash
# build_engine.sh



# Paths
ONNX_MODEL="../models/driver_action.onnx"
ENGINE_FILE="../models/driveraction_fp16.engine"

echo "Building TensorRT engine from ONNX..."
echo "ONNX: $ONNX_MODEL"
echo "ENGINE: $ENGINE_FILE"

# Run TensorRT trtexec
/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$ENGINE_FILE \
    --fp16 \
    --verbose

echo "Engine build complete!"
