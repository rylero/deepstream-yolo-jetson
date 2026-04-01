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
ONNX="$MODEL_DIR/yolo11n.onnx"
ENGINE="$MODEL_DIR/yolo11n_b1_gpu0_fp16.engine"
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

    # Search for the export script — name varies by DeepStream-Yolo version.
    # Also check DeepStream-Yolo/utils directly in case the Dockerfile copy failed.
    EXPORT_SCRIPT=""
    SEARCH_DIRS="$ULTRALYTICS_DIR $DEEPSTREAM_YOLO/utils"
    for dir in $SEARCH_DIRS; do
        for candidate in export_yolo11.py export_yolo.py export_yoloV8.py export_yoloV26.py; do
            if [ -f "$dir/$candidate" ]; then
                # Copy to ultralytics dir if found elsewhere
                [ "$dir" != "$ULTRALYTICS_DIR" ] && cp "$dir/$candidate" "$ULTRALYTICS_DIR/"
                EXPORT_SCRIPT="$candidate"
                break 2
            fi
        done
    done

    if [ -z "$EXPORT_SCRIPT" ]; then
        echo "[export] ERROR: No export script found." >&2
        echo "[export] Searched: $SEARCH_DIRS" >&2
        echo "[export] Available in DeepStream-Yolo/utils:" >&2
        ls "$DEEPSTREAM_YOLO/utils/" >&2
        exit 1
    fi

    echo "[export] Using $EXPORT_SCRIPT"
    # -s 320: use 320x320 input instead of 640x640 — intermediate tensors are 4x smaller,
    # which keeps TRT tactic memory under the ~26 MB CUDA budget available on the Jetson.
    # --dynamic is omitted so TRT builds a fixed-batch engine (simpler, less memory).
    python3 "$EXPORT_SCRIPT" -w yolo11n.pt -s 320 --simplify

    cp yolo11n.onnx "$MODEL_DIR/"
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
