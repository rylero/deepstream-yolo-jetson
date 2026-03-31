# Dockerfile
# DeepStream 7.1 + YOLOv11n + WPILib NetworkTables
# Target: NVIDIA Jetson Orin Nano Super (JetPack 6.1, CUDA 12.6, aarch64)
#
# Build:  docker compose build
# Run:    NT_TEAM_NUMBER=XXXX docker compose up

FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VER=12.6

# ------------------------------------------------------------------ #
# System dependencies                                                  #
# ------------------------------------------------------------------ #
RUN apt-get update && apt-get install -y \
        git \
        cmake \
        build-essential \
        wget \
        unzip \
        pkg-config \
        python3-pip \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-plugins-good \
        gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ #
# Python dependencies for ONNX model export                            #
# ------------------------------------------------------------------ #
RUN pip3 install --no-cache-dir \
        ultralytics \
        onnxslim \
        onnx \
        onnxscript \
        onnxruntime

# ------------------------------------------------------------------ #
# DeepStream-Yolo (custom TensorRT parser + export utilities)          #
# ------------------------------------------------------------------ #
RUN git clone --depth 1 \
        https://github.com/marcoslucianops/DeepStream-Yolo \
        /opt/deepstream/DeepStream-Yolo

# ------------------------------------------------------------------ #
# Ultralytics repo (needed for export_yolo.py)                         #
# ------------------------------------------------------------------ #
RUN git clone --depth 1 \
        https://github.com/ultralytics/ultralytics \
        /opt/ultralytics

# Copy DeepStream-Yolo export scripts; list what's available so failures are visible
RUN echo "[build] DeepStream-Yolo utils contents:" \
    && ls /opt/deepstream/DeepStream-Yolo/utils/ \
    && find /opt/deepstream/DeepStream-Yolo/utils/ -name "export_yolo*.py" \
         -exec cp {} /opt/ultralytics/ \; \
    && echo "[build] Copied export scripts:" \
    && ls /opt/ultralytics/export_yolo*.py 2>/dev/null || echo "[build] WARNING: no export_yolo*.py found"

# ------------------------------------------------------------------ #
# Project directories                                                  #
# ------------------------------------------------------------------ #
RUN mkdir -p \
        /opt/deepstream/models \
        /opt/deepstream/config \
        /opt/deepstream/probe_app

# ------------------------------------------------------------------ #
# Copy project files                                                   #
# ------------------------------------------------------------------ #
COPY config/   /opt/deepstream/config/
COPY probe_app/ /opt/deepstream/probe_app/
COPY scripts/  /opt/deepstream/scripts/

RUN chmod +x /opt/deepstream/scripts/*.sh

# ------------------------------------------------------------------ #
# Build the C probe application                                        #
# ------------------------------------------------------------------ #
RUN cmake \
        -S /opt/deepstream/probe_app \
        -B /opt/deepstream/probe_app/build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/deepstream/probe_app/build --parallel $(nproc)

# ------------------------------------------------------------------ #
# Entrypoint                                                           #
# ------------------------------------------------------------------ #
ENTRYPOINT ["/opt/deepstream/scripts/entrypoint.sh"]
