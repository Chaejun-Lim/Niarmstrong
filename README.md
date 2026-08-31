# Hybrid Runtime (C++ Control + Vision + Python Recorder)

This folder runs the hybrid pipeline where:

- `control_service` handles real serial control (leader/follower) and writes to shared memory.
- `camera_service_cpp` (source: `camera_service_stub.cpp`) handles ZED telemetry and writes to shared memory.
- `latency_monitor.py` monitors control + vision latency in one place.
- `main.py` records episodes using shared-memory action/state.

## 0) Build

```bash
cd hybrid/cpp_pipeline
cmake -S . -B build
cmake --build build -j
```

## 1) Control service (required)

Calibration environment variables are required.

```bash
cd hybrid
LEADER_ZERO_DEG='0,0,0,0,0,0,0' \
LEADER_DIRECTION='1,1,1,1,1,1,1' \
FOLLOWER_ZERO_DEG='0,0,0,0,0,0,0' \
FOLLOWER_MIN_DEG='-90,-90,-90,-90,-90,-90,0' \
FOLLOWER_MAX_DEG='90,90,90,90,90,90,100' \
./cpp_pipeline/build/control_service \
  --shm /hybrid_control \
  --hz 30 \
  --leader-port /dev/ttyACM0 \
  --follower-port /dev/ttyUSB0 \
  --leader-baud 115200 \
  --follower-baud 76800
```

### `control_service` input variables

| Option | Default | Description |
|---|---:|---|
| `--shm` | `/hybrid_control` | POSIX shared memory name for control output |
| `--hz` | `100` | Control loop frequency |
| `--leader-port` | `/dev/ttyACM0` | Leader serial port |
| `--follower-port` | `/dev/ttyUSB0` | Follower serial port |
| `--leader-baud` | `115200` | Leader baud rate |
| `--follower-baud` | `76800` | Follower baud rate (required by follower) |

### Required environment variables (7 comma-separated values each)

| Env | Meaning |
|---|---|
| `LEADER_ZERO_DEG` | Leader zero offsets |
| `LEADER_DIRECTION` | Per-joint direction (`-1` or `1`) |
| `FOLLOWER_ZERO_DEG` | Follower zero offsets |
| `FOLLOWER_MIN_DEG` | Follower joint minimum limits |
| `FOLLOWER_MAX_DEG` | Follower joint maximum limits |

## 2) Vision service (optional, C++)

```bash
cd hybrid/cpp_pipeline
./build/camera_service_cpp \
  --shm /hybrid_vision \
  --fps 30 \
  --depth-mode neural \
  --retrieve-scale 1 \
  --yolo-onnx ../../best.onnx \
  --yolo-conf 0.45
```

### `camera_service_cpp` input variables

| Option | Default | Description |
|---|---:|---|
| `--shm` | `/hybrid_vision` | POSIX shared memory name for vision output |
| `--fps` | `30` | ZED capture FPS |
| `--depth-mode` | `neural` | ZED depth mode (`none/performance/quality/ultra/neural/neural_plus/neural_light`) |
| `--retrieve-scale` | `1.0` | ZED retrieve scale, range `(0, 1]` |
| `--yolo-onnx` | empty | ONNX path for ZED native YOLO-like object detection |
| `--yolo-conf` | `0.25` | Detection confidence threshold `[0, 1]` |

## 3) Latency monitor (recommended)

### A. Subscribe to existing SHM only

```bash
cd hybrid
uv run python3 latency_monitor.py \
  --control-shm /hybrid_control \
  --vision-shm /hybrid_vision
```

### B. Monitor + launch C++ vision service automatically

```bash
cd hybrid
uv run python3 latency_monitor.py \
  --control-shm /hybrid_control \
  --run-cpp-vision \
  --vision-shm /hybrid_vision \
  --camera-service-bin cpp_pipeline/build/camera_service_cpp \
  --fps 30 \
  --zed-depth-mode neural \
  --zed-retrieve-scale 1 \
  --yolo-onnx ../best.onnx \
  --yolo-confidence 0.45
```

### C. Python vision + GUI mode

```bash
cd hybrid
uv run python3 latency_monitor.py \
  --run-python-vision \
  --gui \
  --control-shm /hybrid_control \
  --fps 30 \
  --zed-depth-mode neural \
  --zed-retrieve-scale 1 \
  --sync-yolo
```

### `latency_monitor.py` input variables

| Option | Default | Description |
|---|---:|---|
| `--control-shm` | `/hybrid_control` | Control SHM name |
| `--vision-shm` | `/hybrid_vision` | Vision SHM name |
| `--interval-ms` | `200` | Loop sleep interval |
| `--print-every` | `5` | Print every N samples |
| `--run-python-vision` | off | Run Python ZED+USB+YOLO in-process |
| `--run-cpp-vision` | off | Launch and monitor C++ vision service |
| `--gui` | off | 4-stream GUI (`--run-python-vision` required) |
| `--fps` | `30` | Vision FPS |
| `--camera-index` | `0` | USB camera index |
| `--yolo-weights` | `../best.engine` | Python vision model path |
| `--yolo-onnx` | `../best.onnx` | ONNX path for C++ vision launch |
| `--yolo-confidence` | `0.45` | YOLO confidence |
| `--yolo-device` | `auto` | Python YOLO device |
| `--yolo-quantize` | empty | Precision hint (`16/32/fp16/fp32`) |
| `--yolo-max-det` | `10` | Max detections in Python vision |
| `--sahi` / `--no-sahi` | off | Enable tiled SAHI inference for small objects |
| `--sahi-slice-size` | `320` | Square SAHI tile size in pixels |
| `--sahi-overlap` | `0.2` | Tile overlap ratio |
| `--sync-yolo` / `--no-sync-yolo` | sync on | Sync/asynchronous YOLO execution |
| `--zed-depth-mode` | `neural` | ZED depth mode |
| `--zed-retrieve-scale` | `1.0` | Retrieve scale `(0, 1]` |
| `--depth-min-mm` | `200` | GUI depth colormap minimum |
| `--depth-max-mm` | `2000` | GUI depth colormap maximum |
| `--camera-service-bin` | `cpp_pipeline/build/camera_service_cpp` | C++ vision binary path |
| `--startup-timeout-sec` | `30` | Timeout for first C++ vision frame |
| `--reconnect-ms` | `1000` | SHM reconnect interval |
| `--stale-vision-ms` | `1000` | Stale-vision cutoff |

Output includes `leader_rx_period`, `follower_tx_period`, and `rx_to_tx_delay`.

### Python TensorRT engine runtime

Python vision paths use `../best.engine` by default. When a `.engine` model is
selected, `YoloAnnotator` automatically exposes JetPack's system TensorRT Python
bindings to the `uv` environment, then loads the engine through Ultralytics.
Use `--yolo-weights ../best.onnx` only when an ONNX Runtime path is intended.

### SAHI sliced inference

SAHI is disabled by default because sliced inference runs multiple YOLO
predictions per frame. Enable it for small or distant objects:

```bash
uv run python3 main2.py \
  --repo-id <ACCOUNT/REPO> \
  --task <TASK_NAME> \
  --camera-index 0 \
  --sahi \
  --sahi-slice-size 320 \
  --sahi-overlap 0.2
```

The existing loaded Ultralytics/TensorRT model is reused by SAHI, so enabling
this mode does not load a second model. It does increase inference latency and
GPU/CPU work roughly in proportion to the number of tiles.

The ZED X is retrieved at the resolution selected by the SDK (shown at startup,
normally `1920x1200` for `HD1200`). YOLO and SAHI use that higher-resolution
frame, while `main2.py` downsizes every recorded video stream to `640x480`
when serializing the dataset.

## 4) Camera Preview (`camera_preview.py`)

Run this preview script before recording to check and adjust camera angles for ZED X, USB camera, and YOLO detections.

```bash
cd hybrid
uv run python camera_preview.py --camera-index 2
```

### Options & Controls

- `--camera-index`: USB camera index (default: `0`)
- `--no-yolo`: Disable YOLO detection overlay for lightweight preview
- `--grid-scale`: Scale factor for preview window size (default: `1.0`)

**Key controls in window:**
- `q` / `ESC`: Quit preview
- `y`: Toggle YOLO bounding box display
- `s`: Save frame snapshots to `recordings/snapshots/`

## 5) Recorder (`main.py` / `main2.py`)

```bash
cd hybrid
uv run python3 main.py \
  --repo-id <ACCOUNT/REPO> \
  --task <TASK_NAME> \
  --episodes 5 \
  --duration 10 \
  --fps 30 \
  --control-shm /hybrid_control
```

### `main.py` input variables

| Option | Default | Description |
|---|---:|---|
| `--repo-id` | none | Hugging Face dataset repo (`account/repo`) |
| `--dataset-root` | `recordings` | Local save root |
| `--task` | required | Task name saved in metadata |
| `--episodes` | `5` | Number of episodes |
| `--duration` | `10.0` | Seconds per episode |
| `--fps` | `30` | Recording FPS |
| `--control-shm` | `/hybrid_control` | Control SHM name |
| `--camera-index` | `0` | USB camera index |
| `--yolo-onnx` | `../best.engine` | YOLO model path (legacy option name) |
| `--yolo-weights` | `None` | Deprecated alias for model path |
| `--yolo-confidence` | `0.45` | YOLO confidence `[0,1]` |
| `--yolo-device` | `auto` | Inference device |
| `--yolo-quantize` | empty | Precision hint (`16/32/fp16/fp32`) |
| `--yolo-classes` | empty | Class filter list (`id` or label CSV) |
| `--zed-depth-mode` | `neural` | ZED depth mode |
| `--zed-retrieve-scale` | `1.0` | Retrieve scale `(0, 1]` |
| `--sync-yolo` / `--no-sync-yolo` | sync on | Sync/asynchronous YOLO execution |
| `--push-to-hub` / `--no-push-to-hub` | push on | Upload to Hub after recording |
| `--auto-repo-id` / `--no-auto-repo-id` | off | Auto-prefix short repo name with HF username |

### Recorder environment variables

| Env | Required | Description |
|---|---|---|
| `HF_TOKEN` | only when `--push-to-hub` | Hugging Face write token |

## 5) SHM monitor (simple debug)

```bash
cd hybrid/cpp_pipeline
./build/shm_monitor --ctrl /hybrid_control --vision /hybrid_vision
```

### `shm_monitor` input variables

| Option | Default | Description |
|---|---:|---|
| `--ctrl` | `/hybrid_control` | Control SHM name |
| `--vision` | `/hybrid_vision` | Vision SHM name |

## 6) Pi0-style training and inference (robot-agnostic)

Your `main.py` recorder already writes per-episode `state.npy` and `action.npy`.
The scripts below train and run a compact pi0-style behavior cloning policy
directly from those files, so this works even when your robot is not a native
LeRobot hardware target.

### A. Train from local recordings

```bash
cd hybrid
uv run python3 train_pi0.py \
  --dataset-dir recordings/<ACCOUNT>/<TASK_OR_REPO_NAME> \
  --output-dir outputs/pi0_runs/run1 \
  --epochs 80 \
  --batch-size 256 \
  --lr 3e-4
```

Outputs:
- `policy_best.pt`: best validation checkpoint
- `policy_latest.pt`: latest checkpoint
- `train_metrics.json`: losses and run config

### B. One-shot inference from explicit state

```bash
cd hybrid
uv run python3 infer_pi0.py \
  --checkpoint outputs/pi0_runs/run1/policy_best.pt \
  --state-csv "0,0,0,0,0,0,0"
```

### C. One-shot inference from control shared memory

```bash
cd hybrid
uv run python3 infer_pi0.py \
  --checkpoint outputs/pi0_runs/run1/policy_best.pt \
  --control-shm /hybrid_control \
  --timeout-sec 3
```

Notes:
- No Python/CUDA version changes are required.
- This is an offline imitation baseline using joint state->action.
- For full LeRobot `lerobot-train` with official pi0 configs, first convert to
  official LeRobotDataset v3 (Parquet + MP4 schema expected by LeRobot CLI).
