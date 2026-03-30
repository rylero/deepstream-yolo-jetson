# deepstream-yolo-jetson

DeepStream 7.1 + YOLOv11n + WPILib NetworkTables vision pipeline for NVIDIA Jetson Orin Nano Super. Runs on-robot, publishes detection results to the roboRIO over NT4.

## Target Hardware

- **Device:** NVIDIA Jetson Orin Nano Super
- **JetPack:** 6.1
- **CUDA:** 12.6
- **Architecture:** aarch64

## Project Structure

```
.
├── Dockerfile                  # Multi-stage build: DeepStream + WPILib + probe app
├── docker-compose.yml          # Service definition; mounts camera and model cache
├── .env.example                # Copy to .env and set NT_TEAM_NUMBER
├── config/
│   ├── config_infer_primary_yolo11n.txt  # nvinfer config (FP16, NMS, custom parser)
│   └── labels.txt              # COCO class labels (80 classes)
├── scripts/
│   ├── export_model.sh         # Downloads yolo11n.pt, exports ONNX, builds TRT parser
│   └── entrypoint.sh           # Container entrypoint: runs export then ds_nt_probe
└── probe_app/
    ├── ds_nt_probe.c           # GStreamer + DeepStream pipeline; publishes via NT_RAW
    ├── detection_types.h       # DetectionFrame / Detection structs (packed, fixed-size)
    └── CMakeLists.txt          # cmake build for ds_nt_probe binary
```

## Setup & Usage

### 1. Prerequisites (on the Jetson)

- Docker with NVIDIA Container Runtime installed
- USB camera at `/dev/video0`
- Network connection to the roboRIO (field network or tethered)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your FRC team number:
# NT_TEAM_NUMBER=1234
```

### 3. Build the image

```bash
docker compose build
```

This compiles everything inside the image:
- Installs system deps and Python packages (`ultralytics`, `onnxslim`, `onnx`)
- Clones `DeepStream-Yolo` and `ultralytics` repos
- Downloads WPILib ntcore/wpiutil/wpinet `linuxarm64` shared libs (version pinned via `WPILIB_VER` build arg, default `2025.3.2`)
- Builds `ds_nt_probe` via cmake

### 4. Run

```bash
docker compose up
```

On **first start**, `export_model.sh` runs automatically and:
1. Downloads `yolo11n.pt` from Ultralytics GitHub releases
2. Exports to ONNX using `DeepStream-Yolo`'s export script (`export_yolo.py` or `export_yoloV26.py`)
3. Compiles `libnvdsinfer_custom_impl_Yolo.so` (custom TensorRT parser)
4. TensorRT builds the `.engine` file on first inference (this is the slow step — several minutes)

Results are cached in the `model_cache` Docker volume. Subsequent starts skip all of the above.

### 5. Stopping

```bash
docker compose down
```

The container also handles `SIGINT`/`SIGTERM` for clean GStreamer shutdown.

## Pipeline

```
v4l2src (/dev/video0, 640×480 YUY2 @ 30fps)
  → capsfilter
  → nvvideoconvert
  → nvstreammux (batch-size=1)
  → nvinfer (YOLOv11n, FP16, custom YOLO parser)
  → fakesink
       ^── pad probe → NT_RAW publish on /vision/detections
```

## NetworkTables Output

- **Topic:** `/vision/detections`
- **Type:** `NT_RAW` (raw bytes)
- **Format:** Packed `DetectionFrame` struct (defined in `detection_types.h`)
  - `frame_number` (uint32)
  - `num_detections` (uint32)
  - Up to `MAX_DETECTIONS` × `Detection` entries, each containing:
    - `class_id` (int32), `confidence` (float32)
    - `left`, `top`, `width`, `height` — normalized [0, 1] floats
    - `label` — null-terminated char array

The roboRIO reads this topic and deserializes the struct. The Java-side struct definition must match `detection_types.h` exactly (field order, types, padding).

## Key Configuration

| File | Setting | Default |
|------|---------|---------|
| `docker-compose.yml` | `WPILIB_VER` build arg | `2025.3.2` |
| `config_infer_primary_yolo11n.txt` | `network-mode` | `2` (FP16) |
| `config_infer_primary_yolo11n.txt` | `num-detected-classes` | `80` |
| `config_infer_primary_yolo11n.txt` | `nms-iou-threshold` | `0.45` |
| `config_infer_primary_yolo11n.txt` | `pre-cluster-threshold` | `0.25` |
| `ds_nt_probe.c` | `VIDEO_DEVICE` | `/dev/video0` |
| `ds_nt_probe.c` | `INFER_CONFIG` | `/opt/deepstream/config/config_infer_primary_yolo11n.txt` |

## Updating WPILib Version

Change the `WPILIB_VER` build arg in `docker-compose.yml` to match the current FRC season, then rebuild:

```bash
docker compose build --no-cache
```

## Common Issues

- **TRT engine build hangs:** Normal on first run — building the FP16 engine takes several minutes on the Jetson.
- **Camera not found:** Confirm `/dev/video0` exists on the host before starting. The device is passed through in `docker-compose.yml`.
- **NT not connecting:** Ensure `NT_TEAM_NUMBER` is set correctly in `.env` and the Jetson is on the same network as the roboRIO. The container uses `network_mode: host`.
- **Wrong export script:** `export_model.sh` auto-detects between `export_yolo.py` and `export_yoloV26.py` depending on which commit of DeepStream-Yolo was cloned.
