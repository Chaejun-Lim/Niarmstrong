# Hybrid C++ Pipeline

This folder contains low-latency C++ components for a hybrid architecture:

- `control_service`: real serial control loop (leader/follower) + shared-memory publisher.
- `camera_service_cpp`: real ZED native RGB/depth + native YOLO image publisher using ZED C++ API.
- `shm_monitor`: debug reader for control/vision shared-memory channels.

## Build

```bash
cd hybrid/cpp_pipeline
cmake -S . -B build
cmake --build build -j
```

## Run (C++ services)

Terminal 1:
```bash
./build/control_service \
	--shm /hybrid_control \
	--hz 100 \
	--leader-port /dev/ttyACM0 \
	--follower-port /dev/ttyUSB0 \
	--leader-baud 115200 \
	--follower-baud 76800
```

For autonomous LeRobot rollout, add `--accept-policy`. This creates the
separate `/hybrid_policy` command mapping. While this flag is active, only a
fresh policy command is sent to the follower; a command timeout holds the last
commanded pose instead of falling back to the leader.

```bash
./build/control_service \
    --shm /hybrid_control --hz 100 \
    --leader-port /dev/ttyACM0 --follower-port /dev/ttyUSB0 \
    --leader-baud 115200 --follower-baud 76800 \
    --accept-policy --policy-shm /hybrid_policy --policy-timeout-s 0.25
```

For policy deployment with the leader arm physically disconnected, add
`--policy-only`. It does not open the leader serial port and never forwards a
leader action to the follower:

```bash
./build/control_service \
    --shm /hybrid_control --hz 30 \
    --follower-port /dev/ttyUSB0 --follower-baud 76800 \
    --accept-policy --policy-only --policy-shm /hybrid_policy --policy-timeout-s 0.25 \
    --policy-bootstrap-deg '0,0,0,0,0,0,0'
```

`--policy-bootstrap-deg` is required and must equal the follower's actual
current pose. It is sent only until the first policy action, allowing firmware
that reports state only after a position command to publish initial feedback.

Terminal 2:
```bash
./build/camera_service_cpp \
	--shm /hybrid_vision \
	--fps 30 \
	--depth-mode neural \
	--retrieve-scale 1
```

Terminal 3:
```bash
./build/shm_monitor --ctrl /hybrid_control --vision /hybrid_vision
```

## Next integration steps

1. Add image transport (shared-memory ring buffer) if recorder must avoid direct camera access.
2. Keep Python `hybrid/main.py` for dataset recording and HF upload.

## C++ vision recorder (`main3.py`)

`camera_service_cpp` publishes native-resolution ZED RGB, depth in millimeters,
and an RGB image with native ZED YOLO boxes into `/hybrid_vision`. `main3.py`
reads those images, adds the USB camera, and stores the result using the
existing `640x480` NVENC H.264 recorder. The shared-memory frame sequence is
odd while a frame is being written and even when the frame is complete.

Build after changing the shared-memory contract:

```bash
cmake --build build -j
```

Run from `hybrid/` while `control_service` is already running:

```bash
uv run python main3.py \
	--repo-id <ACCOUNT/REPO> \
	--task "grasp object" \
	--camera-index 0 \
	--yolo-onnx ../best.onnx \
	--episodes 3 \
	--duration 10 \
	--fps 30 \
	--control-shm /hybrid_control \
	--vision-shm /hybrid_vision \
	--no-push-to-hub
```
