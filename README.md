# deepstream-yolo-jetson

FRC vision coprocessor running YOLOv11 on an NVIDIA Jetson Orin Nano Super. Publishes ball detections as ground-plane coordinates to the roboRIO over NetworkTables 4.

## Hardware

| Component | Details |
|-----------|---------|
| Device | NVIDIA Jetson Orin Nano Super |
| JetPack | 6.1 |
| CUDA | 12.6 |
| Cameras | 2× Arducam OV9782 USB (1280×720 @ 120 fps MJPEG) |

## How it works

```
v4l2src (1280×720 MJPEG)
  → jpegdec → videoconvert → nvvideoconvert
  → nvstreammux (batch=2, 640×360)
  → nvinfer (YOLOv11, FP16, custom YOLO parser)
       └── pad probe → project bbox → NT4 publish
  → nvmultistreamtiler → nvdsosd
  → tee ┬→ fakesink
        └→ jpegenc → appsink → HTTP :8080 (MJPEG stream)
```

Each detection's bounding box centre is back-projected to the floor plane (z = 0) using the calibrated camera intrinsics and known mounting geometry, producing metric `(x, y, distance)` coordinates. See [NT_FORMAT.md](NT_FORMAT.md) for the full NetworkTables schema.

## Setup

### 1. Prerequisites (on the Jetson)

- Docker with NVIDIA Container Runtime
- Two USB cameras at `/dev/video0` and `/dev/video2`
- Network connection to the roboRIO

### 2. Configure

```bash
cp .env.example .env
# Set your team number:
# NT_TEAM_NUMBER=1234
```

Camera mounting parameters (defaults: 0.5 m height, 20° downward pitch):

```bash
# In .env or docker-compose environment section:
CAM0_HEIGHT=0.52
CAM0_PITCH_DEG=22.5
CAM1_HEIGHT=0.52
CAM1_PITCH_DEG=22.5
```

### 3. Place your model

Copy your trained `.pt` file to the project root:

```bash
cp path/to/your-model.pt 640-yolo11s-fp32.pt
```

The `docker-compose.yml` mounts this as `custom_model.pt` inside the container. The model must be a YOLOv11 detector exported at 640×640.

### 4. Build

```bash
docker compose build
```

Compiles inside the image:
- Python packages (`ultralytics`, `onnxslim`, `onnx`)
- DeepStream-Yolo custom TensorRT parser
- `ds_nt_probe` C application (GStreamer + NT4)

### 5. Run

```bash
docker compose up
```

**First start** runs `export_model.sh` automatically:
1. Exports `.pt` → ONNX via DeepStream-Yolo's export script
2. Compiles `libnvdsinfer_custom_impl_Yolo.so`
3. Builds the FP16 TensorRT engine with `trtexec` — **takes several minutes**

Subsequent starts skip all of the above (results cached in the `model_cache` Docker volume).

### 6. Stop

```bash
docker compose down
```

## NetworkTables Output

See [NT_FORMAT.md](NT_FORMAT.md) for full documentation.

| Topic | Type | Description |
|-------|------|-------------|
| `objectdetection/balls/0` | `struct:BallDetection[]` | Detections from left camera |
| `objectdetection/balls/1` | `struct:BallDetection[]` | Detections from right camera |
| `objectdetection/heartbeat` | `int` | Increments ~30×/sec; use to detect a dead pipeline |

Each `BallDetection` contains `double x, y, distance` in metres, camera-relative.

**Java subscriber:**

```java
StructArraySubscriber<BallDetection> sub = NetworkTableInstance.getDefault()
    .getStructArrayTopic("objectdetection/balls/0", BallDetection.struct)
    .subscribe(new BallDetection[]{});
```

## Debug stream

Tiled MJPEG stream with bounding boxes visible at:

```
http://<jetson-ip>:8080/
```

Includes a camera control panel for brightness, exposure, and white balance.

## Camera calibration

Intrinsics are loaded from `L.json` (cam 0) and `R.json` (cam 1) — OpenCV calibration files for the Arducam OV9782 at 1280×720. To recalibrate, replace these files and rebuild the image.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NT_TEAM_NUMBER` | `0` | FRC team number |
| `NUM_CAMERAS` | `2` | Number of USB cameras |
| `CAMERA_DEVICES` | `/dev/video0,/dev/video2` | Comma-separated capture device paths |
| `VIDEO_URI` | *(unset)* | Set to a file/RTSP URI to use instead of cameras |
| `CAM0_HEIGHT` | `0.5` | Left camera height above floor (m) |
| `CAM0_PITCH_DEG` | `20.0` | Left camera tilt below horizontal (°) |
| `CAM1_HEIGHT` | `0.5` | Right camera height above floor (m) |
| `CAM1_PITCH_DEG` | `20.0` | Right camera tilt below horizontal (°) |

## Project structure

```
├── Dockerfile                          Multi-stage build
├── docker-compose.yml                  Service definition
├── .env.example                        Copy to .env, set NT_TEAM_NUMBER
├── L.json / R.json                     Camera calibration (OV9782, 1280×720)
├── NT_FORMAT.md                        NetworkTables wire format documentation
├── config/
│   ├── config_infer_primary_yolo11n.txt  nvinfer config (FP16, NMS, custom parser)
│   └── labels.txt                      Class labels
├── scripts/
│   ├── export_model.sh                 ONNX export + TRT engine build (idempotent)
│   └── entrypoint.sh                   Container entrypoint
└── probe_app/
    ├── ds_nt_probe.c                   GStreamer pipeline + NT4 publisher
    ├── detection_types.h               Internal detection structs
    └── CMakeLists.txt                  cmake build
```

## Common issues

**TRT engine build hangs** — normal on first run. Building the FP16 engine takes several minutes on the Jetson. Wait it out.

**Camera not found** — run `v4l2-ctl --list-devices` on the Jetson to confirm device paths, then set `CAMERA_DEVICES` accordingly.

**NT not connecting** — confirm `NT_TEAM_NUMBER` is correct and the Jetson is on the same network as the roboRIO. The container uses `network_mode: host`.

**Wrong detection coordinates** — measure and set `CAM0_HEIGHT` and `CAM0_PITCH_DEG` accurately for your robot's camera mounting.
