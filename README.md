# Alpamayo 1.5 ROS 2 Node Usage Guide

![Alpamayo Autoware Demo](images/alpamayo-autoware.gif)

This guide explains how to set up and run the Alpamayo ROS 2 node.

## Prerequisites

| Requirement | Specification                                |
| ----------- | -------------------------------------------- |
| **Python**  | 3.10.x (for compatibility with ROS 2 Humble) |
| **ROS 2**   | Humble (must be installed)                   |
| **GPU**     | NVIDIA GPU (24 GB+ VRAM recommended)         |
| **OS**      | Linux (tested)                               |

## Setup Instructions

### 1. Install uv

If not already installed, install uv using the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Create Virtual Environment with Python 3.10

**Important**: You must use Python 3.10 for compatibility with ROS 2 Humble.

Remove any existing venv and recreate it with Python 3.10:

```bash
# Remove existing venv (if it exists)
rm -rf a1_5_venv

# Create new venv with Python 3.10
uv venv a1_5_venv --python python3.10

# Activate the virtual environment
source a1_5_venv/bin/activate

# Install dependencies
uv sync --active
```

### 3. HuggingFace Authentication

Request access to the Alpamayo model and dataset:

- [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Once access is granted, authenticate using the HuggingFace CLI:

```bash
# Install HuggingFace Hub (if not already installed)
pip install huggingface_hub

# Login with your token
huggingface-cli login
```

You can obtain your access token at: <https://huggingface.co/settings/tokens>

## Running the ROS 2 Node

### Method 1: Direct Script Execution (Recommended)

Source the ROS 2 environment and run the node using Python from the virtual environment:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source Autoware environment (need to change correct path)
source ~/workspace/autoware/install/setup.bash

# Activate virtual environment
source a1_5_venv/bin/activate

# Run the node
python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
  -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', '/sensing/camera/camera1/image_raw/compressed', '/sensing/camera/camera4/image_raw/compressed', '/sensing/camera/camera2/image_raw/compressed']" \
  -p camera_indices:="[0, 1, 2, 6]"

# Run the node (rosbag mode)
# python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
#   -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', '/sensing/camera/camera1/image_raw/compressed', '/sensing/camera/camera4/image_raw/compressed', '/sensing/camera/camera2/image_raw/compressed']" \
#   -p camera_indices:="[0, 1, 2, 6]" \
#   -p use_sim_time:=true

```

### Method 2: Using colcon Build

If you want to build as a ROS 2 package using colcon:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source Autoware environment (need to change correct path)
source ~/workspace/autoware/install/setup.bash
# Activate virtual environment
source a1_5_venv/bin/activate

# Build the package
colcon build --packages-select alpamayo_ros --symlink-install

# Source the workspace
source install/setup.bash

# Run the node
python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
  -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', '/sensing/camera/camera1/image_raw/compressed', '/sensing/camera/camera4/image_raw/compressed', '/sensing/camera/camera2/image_raw/compressed']" \
  -p camera_indices:="[0, 1, 2, 6]"

# Run the node (rosbag mode)
# python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
#   -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', '/sensing/camera/camera1/image_raw/compressed', '/sensing/camera/camera4/image_raw/compressed', '/sensing/camera/camera2/image_raw/compressed']" \
#   -p camera_indices:="[0, 1, 2, 6]" \
#   -p use_sim_time:=true

```

### Method 3: Using Launch File

If a launch file is available:

```bash
# Source ROS 2 environment and workspace
source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

# Source Autoware environment (need to change correct path)
source ~/workspace/autoware/install/setup.bash

# Run the launch file
ros2 launch alpamayo_ros alpamayo.launch.py
```

## Parameters

The Alpamayo node can be configured with the following ROS parameters:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `camera_topics` | (required) | List of camera image topics (CompressedImage type) |
| `camera_indices` | (required) | Camera index for each topic (0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto) |
| `odometry_topic` | `/localization/kinematic_state` | Odometry topic |
| `route_topic` | `/planning/mission_planning/route` | Route topic (optional — nav-text prompt) |
| `trajectory_topic` | `/alpamayo/predicted_trajectory` | Output topic for predicted trajectory |
| `cot_topic` | `/alpamayo/reasoning` | Output topic for reasoning trace |
| `cot_with_stamped_topic` | `/alpamayo/reasoning_stamped` | Output topic for timestamped reasoning trace |
| `nav_text_topic` | `/alpamayo/nav_text` | Output topic for navigation-text (debug) |
| `lanelet2_map_path` | `""` | Optional Lanelet2 OSM map for nav-text grounding |
| `inference_period_sec` | `0.1` | Inference execution period (seconds) |
| `num_diffusion_steps` | `5` | Flow-matching Euler steps (5 keeps ADE within ~1% of 10) |
| `use_greedy_decode` | `true` | Greedy sampling; bypasses `top_p` / `temperature` |
| `top_p` | `0.98` | Nucleus sampling cutoff (ignored when greedy) |
| `temperature` | `0.6` | Sampling temperature (ignored when greedy) |
| `max_generation_length` | `64` | VLM token budget per tick |
| `use_sim_time` | `false` | Whether to use simulation time |

## Troubleshooting

### Python Version Mismatch

If you see error message `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`:

- Cause: The venv was created with a Python version other than 3.10
- Solution: Follow step 2 above to recreate the venv with Python 3.10

### CUDA Out-of-Memory Errors

If you encounter memory errors:

1. Ensure you're using a GPU with at least 24 GB VRAM
2. Close other GPU-intensive applications
3. Increase the inference period (`inference_period_sec`)

### Flash Attention Issues

If you encounter compatibility issues with Flash Attention 2, you can use an alternative implementation in the model code:

```python
config.attn_implementation = "sdpa"
```

### Slow Model Download

On first run, the model weights (approximately 22 GB) will be downloaded. This can take time depending on your connection speed (approximately 2.5 minutes on a 100 MB/s connection).

## Output Topics

The node publishes the following topics:

- `/alpamayo/predicted_trajectory` (autoware_planning_msgs/Trajectory): Predicted vehicle trajectory
- `/alpamayo/reasoning` (std_msgs/String): Chain-of-Causation reasoning text
- `/alpamayo/reasoning_stamped` (autoware_internal_debug_msgs/StringStamped): Timestamped reasoning text
- `/alpamayo/predicted_trajectory_markers` (visualization_msgs/MarkerArray): Visualization markers for RViz

## FlashDrive Acceleration

`src/alpamayo1_5/flashdrive/` ports the [FlashDrive](https://z-lab.ai/projects/flashdrive/)
algorithm-system co-design to Alpamayo 1.5. Exposed as a **Python library**
(see usage below); the ROS node runs the unaccelerated baseline unless you
wire `apply_flashdrive(...)` into your own entry point.

Measured on an RTX PRO 6000 (Blackwell, sm_120) with multi-camera rosbag
frames and an 8-step streaming window:

| Config                             | p50 (ms) | best step (ms) | speedup |
| ---------------------------------- | -------- | -------------- | ------- |
| baseline (upstream, 10-step diff)  | 711      | 669            | 1.00×   |
| FlashDrive (5-step diff + stack)   | 292      | 146            | 2.44×   |

Baseline runs the upstream inference path with `num_inference_steps=10`
(FlowMatching's default). FlashDrive cuts that to 5 Euler steps via
`adaptive_flow` and adds the four remaining toggles; reproducing the
table is a single command (see below).

### Prerequisites for the bench

Install the `bench` extra (adds the `rosbags` reader needed to stream
real camera frames):

```bash
uv sync --active --group bench
```

The reproduction uses the FlashDrive-finetuned checkpoint (the original
`nvidia/Alpamayo-1.5-10B` was not exposed to streaming caches during
training, so the `streaming_lm` / `streaming_vision` approximations drift
on it). Download from the [FlashDriveVLA HF org](https://huggingface.co/FlashDriveVLA):

```bash
huggingface-cli download FlashDriveVLA/Alpamayo-1.5-10B-finetuned \
  --local-dir models/Alpamayo-1.5-10B-finetuned
```

### Applying the full stack in Python

```python
import torch
from accelerate import init_empty_weights
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.flashdrive import FlashDriveConfig, apply_flashdrive
from alpamayo1_5.flashdrive.load_helpers import (
    load_full_state_dict, patch_alpamayo15_config_for_qwen3vl,
)

teacher = "models/Alpamayo-1.5-10B-finetuned"

patch_alpamayo15_config_for_qwen3vl(teacher)
cfg = Alpamayo1_5Config.from_pretrained(teacher)
with init_empty_weights(include_buffers=False):
    model = Alpamayo1_5(cfg)
model = model.to(torch.bfloat16)
model.load_state_dict(load_full_state_dict(teacher), strict=False, assign=True)
model = model.to("cuda").eval()
model.diffusion.num_inference_steps = 5

fd = FlashDriveConfig.from_strings([
    "adaptive_flow",
    "kernel_fusion_qkv", "kernel_fusion_mlp",
    "streaming_vision", "streaming_lm",
])
apply_flashdrive(model, fd)
```

Five toggles are independent — drop any to A/B isolate contributions.

### Reproducing the bench

```bash
PYTHONPATH=src python scripts/bench_streaming_e2e.py \
  --teacher models/Alpamayo-1.5-10B-finetuned \
  --bag     /path/to/rosbag \
  --steps   8 --max-gen-tokens 16
```

Baseline uses 10 diffusion steps (upstream default, `--baseline-diffusion-steps`);
FlashDrive configs use 5 (`--diffusion-steps`). Both are run in the same
invocation for a fair side-by-side. The script streams real JPEG frames
from the rosbag (`/sensing/camera/camera{0,1,2,7}/image_raw/compressed`
topics — see `DEFAULT_CAMERA_TOPICS` in the script) and reports per-step
p50 / best / worst wall-clock ms under each config.

Further bench harnesses (stage-level, synthetic-input — handy for
isolating per-Linear speedups without a rosbag handy):

- `scripts/bench_flashdrive.py` — per-config A/B table
- `scripts/bench_streaming_vision.py` — ViT-cache ablation
- `scripts/bench_stage_breakdown.py` — per-stage latency breakdown

## License and Disclaimer

- Inference code: Apache License 2.0
- Model weights: Non-commercial license

The FlashDrive acceleration stack in `src/alpamayo1_5/flashdrive/` is a
re-implementation of ideas from [Z Lab's FlashDrive](https://z-lab.ai/projects/flashdrive/).

For details, see the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B).

Alpamayo 1.5 is a pre-trained reasoning model for research purposes and is not
a complete autonomous driving stack. It is not intended for use in production
environments.
