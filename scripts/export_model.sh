#!/bin/bash
# export_model.sh
#
# Converts the custom model (mounted at /opt/deepstream/models/custom_model.pt)
# to ONNX via DeepStream-Yolo's export script, and compiles the custom TensorRT
# parser library (nvdsinfer_custom_impl_Yolo).
#
# Skips work that has already been done (idempotent).
# Called by entrypoint.sh on every container start.

set -euo pipefail

MODEL_DIR=/opt/deepstream/models
CUSTOM_PT="$MODEL_DIR/custom_model.pt"
ONNX="$MODEL_DIR/custom_yolo11n_640.onnx"
ENGINE="$MODEL_DIR/custom_yolo11n_640_b1_gpu0_fp16.engine"
DEEPSTREAM_YOLO=/opt/deepstream/DeepStream-Yolo
ULTRALYTICS_DIR=/opt/ultralytics
SO="$DEEPSTREAM_YOLO/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"

export CUDA_VER=12.6

# ------------------------------------------------------------------ #
# 1. Export ONNX model                                                 #
# ------------------------------------------------------------------ #
if [ ! -f "$ONNX" ]; then
    if [ ! -f "$CUSTOM_PT" ]; then
        echo "[export] ERROR: Custom model not found at $CUSTOM_PT" >&2
        echo "[export] Ensure 640-yolo11n-fp32.pt is in the project root and" >&2
        echo "[export] the docker-compose.yml volume mount is present." >&2
        exit 1
    fi

    echo "[export] Exporting ONNX from custom model: $CUSTOM_PT"
    cd "$ULTRALYTICS_DIR"

    # Copy model to ultralytics working dir
    cp "$CUSTOM_PT" ./custom_model.pt

    # Search for the export script — name varies by DeepStream-Yolo version.
    EXPORT_SCRIPT=""
    SEARCH_DIRS="$ULTRALYTICS_DIR $DEEPSTREAM_YOLO/utils"
    for dir in $SEARCH_DIRS; do
        for candidate in export_yolo11.py export_yolo.py export_yoloV8.py export_yoloV26.py; do
            if [ -f "$dir/$candidate" ]; then
                [ "$dir" != "$ULTRALYTICS_DIR" ] && cp "$dir/$candidate" "$ULTRALYTICS_DIR/"
                EXPORT_SCRIPT="$candidate"
                break 2
            fi
        done
    done

    if [ -z "$EXPORT_SCRIPT" ]; then
        echo "[export] ERROR: No export script found." >&2
        echo "[export] Available in DeepStream-Yolo/utils:" >&2
        ls "$DEEPSTREAM_YOLO/utils/" >&2
        exit 1
    fi

    echo "[export] Using $EXPORT_SCRIPT"
    # -s 320: 320×320 inference input — 4× smaller intermediate tensors than 640,
    # keeping TRT tactic memory within Jetson's ~26 MB CUDA budget.
    python3 "$EXPORT_SCRIPT" -w custom_model.pt -s 640 --simplify

    cp custom_model.onnx "$ONNX"
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

# ------------------------------------------------------------------ #
# 3. Pre-build TRT engine with trtexec                                 #
#                                                                      #
# This runs BEFORE ds_nt_probe (and before GStreamer/DeepStream        #
# allocate NVMM buffers), so CUDA has far more free memory available   #
# for tactic selection. When nvinfer finds the engine file on disk it  #
# loads it directly without rebuilding.                                #
# ------------------------------------------------------------------ #
if [ ! -f "$ENGINE" ]; then
    echo "[export] Building TRT FP16 engine with trtexec (this takes several minutes)..."

    # Locate trtexec — path varies by TRT/JetPack version
    TRTEXEC=""
    for candidate in \
        /usr/src/tensorrt/bin/trtexec \
        /usr/bin/trtexec \
        /opt/tensorrt/bin/trtexec; do
        if [ -x "$candidate" ]; then
            TRTEXEC="$candidate"
            break
        fi
    done

    if [ -z "$TRTEXEC" ]; then
        echo "[export] WARNING: trtexec not found; nvinfer will build the engine on first run."
    else
        export CUDA_VER=12.6
        "$TRTEXEC" \
            --onnx="$ONNX" \
            --saveEngine="$ENGINE" \
            --fp16 \
            --memPoolSize=workspace:256 \
            --verbose 2>&1 | grep -E '^\[|^&|\[export\]|TRT-|error|Error|warning|Warning|Building|Completed' || true

        if [ -f "$ENGINE" ]; then
            echo "[export] TRT engine built successfully: $ENGINE"
        else
            echo "[export] WARNING: trtexec did not produce an engine; nvinfer will rebuild on first run."
        fi
    fi
else
    echo "[export] TRT engine already exists, skipping trtexec build."
fi

echo "[export] Setup complete."
