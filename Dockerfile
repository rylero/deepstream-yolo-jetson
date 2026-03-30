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
        gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ #
# Python dependencies for ONNX model export                            #
# ------------------------------------------------------------------ #
RUN pip3 install --no-cache-dir \
        ultralytics \
        onnxslim \
        onnx

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
        /opt/ultralytics \
    && cp /opt/deepstream/DeepStream-Yolo/utils/export_yolo*.py \
          /opt/ultralytics/ 2>/dev/null || true

# ------------------------------------------------------------------ #
# WPILib ntcore + wpiutil + wpinet (linuxarm64)                         #
# Update WPILIB_VER to match your FRC season if needed.                #
# ------------------------------------------------------------------ #
ARG WPILIB_VER=2025.3.2
ARG WPILIB_BASE=https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first

RUN for lib in ntcore/ntcore-cpp wpiutil/wpiutil-cpp wpinet/wpinet-cpp; do \
        name=$(basename "$lib"); \
        \
        # Shared libraries (platform-specific)
        so_url="${WPILIB_BASE}/${lib}/${WPILIB_VER}/${name}-${WPILIB_VER}-linuxarm64.zip"; \
        echo "[wpilib] Downloading $so_url"; \
        wget -q -O "/tmp/${name}-so.zip" "$so_url"; \
        unzip -q "/tmp/${name}-so.zip" -d "/tmp/${name}-so"; \
        find "/tmp/${name}-so" -maxdepth 4 -name "*.so*" \
            -exec cp -P {} /usr/local/lib/ \; ; \
        rm -rf "/tmp/${name}-so" "/tmp/${name}-so.zip"; \
        \
        # Headers (platform-independent, separate artifact)
        hdr_url="${WPILIB_BASE}/${lib}/${WPILIB_VER}/${name}-${WPILIB_VER}-headers.zip"; \
        echo "[wpilib] Downloading $hdr_url"; \
        wget -q -O "/tmp/${name}-hdr.zip" "$hdr_url"; \
        unzip -q "/tmp/${name}-hdr.zip" -d "/tmp/${name}-hdr"; \
        cp -r "/tmp/${name}-hdr/." /usr/local/include/; \
        rm -rf "/tmp/${name}-hdr" "/tmp/${name}-hdr.zip"; \
    done \
    && ldconfig

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
