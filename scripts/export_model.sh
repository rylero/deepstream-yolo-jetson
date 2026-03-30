#!/bin/bash
# export_model.sh
#
# Downloads yolo11n.pt, converts to ONNX via DeepStream-Yolo's export script,
# and compiles the custom TensorRT parser library (nvdsinfer_custom_impl_Yolo).
#
# Skips work that has already been done (idempotent).
# Called by entrypoint.sh on every container start.

set -euo pipefail

MODEL_DIR=/opt/deepstream/models
ONNX="$MODEL_DIR/yolo11n.pt.onnx"
DEEPSTREAM_YOLO=/opt/deepstream/DeepStream-Yolo
ULTRALYTICS_DIR=/opt/ultralytics
SO="$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"

export CUDA_VER=12.6

# ------------------------------------------------------------------ #
# 1. Export ONNX model                                                 #
# ------------------------------------------------------------------ #
if [ ! -f "$ONNX" ]; then
    echo "[export] Downloading yolo11n.pt..."
    wget -q --show-progress \
        -O "$ULTRALYTICS_DIR/yolo11n.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"

    echo "[export] Exporting ONNX..."
    cd "$ULTRALYTICS_DIR"

    # The export script may be named export_yolo.py or export_yoloV26.py
    # depending on which commit of DeepStream-Yolo was cloned.
    EXPORT_SCRIPT=""
    for candidate in export_yolo.py export_yoloV26.py; do
        if [ -f "$ULTRALYTICS_DIR/$candidate" ]; then
            EXPORT_SCRIPT="$candidate"
            break
        fi
    done

    if [ -z "$EXPORT_SCRIPT" ]; then
        echo "[export] ERROR: No export script found in $ULTRALYTICS_DIR" >&2
        echo "[export] Expected export_yolo.py or export_yoloV26.py" >&2
        exit 1
    fi

    echo "[export] Using $EXPORT_SCRIPT"
    python3 "$EXPORT_SCRIPT" -w yolo11n.pt --simplify --dynamic

    cp yolo11n.pt.onnx "$MODEL_DIR/"
    # Copy labels.txt only if it doesn't already exist in config
    if [ -f "labels.txt" ]; then
        cp labels.txt "$MODEL_DIR/"
    fi

    echo "[export] ONNX export complete: $ONNX"
else
    echo "[export] ONNX already exists, skipping export."
fi

# ------------------------------------------------------------------ #
# 2. Compile nvdsinfer_custom_impl_Yolo                                #
# ------------------------------------------------------------------ #
if [ ! -f "$SO" ]; then
    echo "[export] Compiling nvdsinfer_custom_impl_Yolo (CUDA_VER=$CUDA_VER)..."
    make -C "$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo" clean
    make -C "$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo"
    echo "[export] Custom parser library built: $SO"
else
    echo "[export] Custom parser library already exists, skipping compile."
fi

echo "[export] Setup complete."
