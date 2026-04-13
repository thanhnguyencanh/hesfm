# HESFM: Hierarchical Evidential Semantic-Functional Mapping

[![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)

HESFM is a ROS Noetic package for uncertainty-aware semantic mapping from RGB-D input. It combines semantic segmentation, semantic point cloud generation, Gaussian primitive aggregation, and evidential map updates into a navigation-oriented pipeline for indoor robots.

<p align="center">
  <img src="docs/images/hesfm_overview.png" alt="HESFM Overview" width="800"/>
</p>

## Highlights

- Real-time semantic mapping pipeline for Intel RealSense RGB-D cameras
- Uncertainty-aware map updates with semantic, spatial, observation, and temporal terms
- Gaussian primitive abstraction for denser but more stable map updates
- Zero-copy semantic cloud to mapper path via ROS nodelets
- YAML-backed runtime parameters for semantic cloud and mapper
- Ready-to-run tmux launcher for split camera / segmentation / mapper / RViz workflows

## Repository Scope

This repository contains the `hesfm` ROS package plus several sibling research codebases kept for reference. The package you build and run here is:

- `hesfm/`

The other top-level directories such as `DFormer/`, `ESANet/`, `SLIDE_SLAM/`, `EvSemMap/`, and similar folders are not launched by default from the HESFM package.

## Requirements

### Software

- Ubuntu 20.04
- ROS Noetic
- C++17 toolchain
- Python 3.8+
- OpenCV
- PCL
- Eigen
- `nodelet`, `cv_bridge`, `image_transport`, `tf2_ros`, `dynamic_reconfigure`

### Hardware

- RGB-D camera: Intel RealSense D455 recommended
- GPU for segmentation:
  - Desktop: NVIDIA RTX-class GPU for PyTorch or TensorRT workflows
  - Jetson: Xavier AGX / Orin AGX with TensorRT-backed ESANet recommended

## Installation

### 1. Create a catkin workspace

```bash
mkdir -p ~/sslam_ws/src
cd ~/sslam_ws/src
git clone <your-hesfm-repository-url> hesfm
cd ..
```

### 2. Install ROS dependencies

```bash
sudo apt update
sudo apt install \
  ros-noetic-desktop-full \
  ros-noetic-realsense2-camera \
  ros-noetic-realsense2-description \
  ros-noetic-pcl-ros \
  ros-noetic-cv-bridge \
  ros-noetic-tf2-ros \
  ros-noetic-tf2-sensor-msgs \
  ros-noetic-image-transport \
  ros-noetic-dynamic-reconfigure \
  ros-noetic-nodelet
```

### 3. Install system libraries

```bash
sudo apt install \
  libeigen3-dev \
  libpcl-dev \
  libopencv-dev \
  libboost-all-dev
```

### 4. Resolve package dependencies

```bash
cd ~/sslam_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 5. Build the workspace

```bash
cd ~/sslam_ws
catkin build hesfm
source devel/setup.bash
```

To auto-source on new terminals:

```bash
echo "source ~/sslam_ws/devel/setup.bash" >> ~/.bashrc
```

## Models

The segmentation node supports multiple backends, but the current launch files are centered on:

- `esanet_trt` for Jetson / TensorRT deployment
- `esanet_pytorch` for desktop development
- `dformer` for higher-accuracy experiments

Place the model files under [`models/`](models) and update launch arguments if you use custom paths. The default launch configuration expects files such as:

- `models/esanet_r34_sunrgbd_fp16.engine`
- `models/esanet_r34_sunrgbd.pth`
- `models/dformer_large_sunrgbd.pth`

## Quick Start

### Full pipeline

This launches camera, segmentation, semantic cloud, mapper, optional OpenVINS, and RViz from one launch file:

```bash
roslaunch hesfm hesfm.launch
```

Useful overrides:

```bash
roslaunch hesfm hesfm.launch rviz:=false
roslaunch hesfm hesfm.launch segmentation_backend:=dformer
roslaunch hesfm hesfm.launch launch_ov_msckf:=false
roslaunch hesfm hesfm.launch use_nodelets:=true
```

### Segmentation only

```bash
roslaunch hesfm segmentation_only.launch
```

Run against an already-running camera:

```bash
roslaunch hesfm segmentation_only.launch launch_camera:=false
```

### Mapper only

This starts semantic cloud plus mapper without starting camera or segmentation. It is intended for split workflows where those are launched separately.

```bash
roslaunch hesfm mapper_only.launch
```

### Navigation and exploration

```bash
roslaunch hesfm hesfm_navigation.launch
roslaunch hesfm hesfm_exploration.launch
```

## tmux Workflow

To run HESFM in separate panes like the SLIDE_SLAM workflow, use:

```bash
bash "$(rospack find hesfm)/scripts/tmux_hesfm_pipeline.sh"
```

The script creates:

- `Core` window: `roscore`
- `Main` window:
  - camera
  - segmentation
  - semantic cloud + mapper
  - RViz
- `Kill` window: session shutdown helper

Example overrides:

```bash
LAUNCH_OV_MSCKF=false LAUNCH_RVIZ=false bash "$(rospack find hesfm)/scripts/tmux_hesfm_pipeline.sh"
```

```bash
USE_NODELETS=false SEGMENTATION_BACKEND=dformer bash "$(rospack find hesfm)/scripts/tmux_hesfm_pipeline.sh"
```

## Configuration

### YAML vs launch arguments

Runtime mapping parameters are loaded from [`config/hesfm_params.yaml`](config/hesfm_params.yaml).

This YAML is used by:

- `semantic_cloud_node` / `SemanticCloudNodelet`
- `hesfm_mapper_node` / `HESFMMapperNodelet`

Launch files still control topology and feature switches such as:

- `launch_camera`
- `launch_segmentation`
- `semantic_cloud`
- `launch_mapper`
- `use_nodelets`
- `launch_ov_msckf`
- `rviz`

That means:

- Use the YAML file for mapper / cloud runtime behavior
- Use launch arguments for what processes start and how the system is composed

### Important YAML sections

[`config/hesfm_params.yaml`](config/hesfm_params.yaml) is organized into:

- `map`: voxel resolution and map bounds
- `uncertainty`: uncertainty weights and neighborhood settings
- `primitives`: primitive target count and incremental-update options
- `kernel`: adaptive kernel behavior
- `navigation`: relevant and traversable semantic classes
- `processing`: queue sizes and publish rates
- `sensor`: `min_range` and `max_range`
- `semantic_cloud`: output frame and uncertainty defaults

Example:

```yaml
sensor:
  min_range: 0.1
  max_range: 6.0

primitives:
  use_incremental_updates: true
  incremental_max_distance: 0.5
  full_rebuild_interval: 0
```

### Current parameter behavior

- `sensor.max_range` is the active runtime range limit for semantic cloud filtering and mapper uncertainty handling
- The old `max_depth` naming is no longer the intended config path
- If the same setting exists in both YAML and a launch file, the explicit launch value wins

## Launch Files

### Core launch files

- [`launch/hesfm.launch`](launch/hesfm.launch): full pipeline
- [`launch/segmentation_only.launch`](launch/segmentation_only.launch): segmentation-only workflow
- [`launch/mapper_only.launch`](launch/mapper_only.launch): semantic cloud + mapper only
- [`launch/d455_camera.launch`](launch/d455_camera.launch): RealSense D455 setup

### Application launch files

- [`launch/hesfm_navigation.launch`](launch/hesfm_navigation.launch): navigation-oriented stack
- [`launch/hesfm_exploration.launch`](launch/hesfm_exploration.launch): exploration setup
- [`launch/hesfm_evaluation.launch`](launch/hesfm_evaluation.launch): evaluation utilities

## Runtime Notes for Jetson Xavier AGX

For better real-time performance on Xavier AGX:

- Use `segmentation_backend:=esanet_trt`
- Keep `use_nodelets:=true` so semantic cloud and mapper share `PointCloud2` zero-copy
- Disable RViz during deployment with `rviz:=false`
- Leave debug publishers off unless you are actively inspecting them:
  - `publish_primitives:=false`
  - `publish_uncertainty_map:=false`
  - `publish_uncertainty_cloud:=false`
- Keep incremental primitive updates enabled in YAML:
  - `primitives.use_incremental_updates: true`
- Tune `sensor.max_range` conservatively for indoor scenes to reduce per-frame load
- Lower `processing.downsample_factor` only if you truly need denser clouds

## Package Layout

The main package directories are:

```text
hesfm/
|-- include/hesfm/        C++ public headers
|-- src/hesfm_core/       Core mapping library
|-- src/nodes/            Standalone ROS nodes
|-- src/nodelets/         Zero-copy nodelet implementations
|-- scripts/              Python nodes and utilities
|-- launch/               Launch files
|-- config/               YAML runtime configuration
|-- rviz/                 RViz configurations
|-- msg/                  Custom ROS messages
|-- srv/                  Custom ROS services
|-- docs/                 Images and documentation assets
`-- test/                 Tests
```

## Main Components

### Semantic segmentation

[`scripts/semantic_segmentation_node.py`](scripts/semantic_segmentation_node.py) consumes RGB-D input and publishes semantic outputs for the rest of the pipeline.

### Semantic cloud

[`src/nodes/semantic_cloud_node.cpp`](src/nodes/semantic_cloud_node.cpp) and [`src/nodelets/semantic_cloud_nodelet.cpp`](src/nodelets/semantic_cloud_nodelet.cpp) convert depth plus semantic predictions into a semantic point cloud.

### Mapper

[`src/nodes/hesfm_mapper_node.cpp`](src/nodes/hesfm_mapper_node.cpp) and [`src/nodelets/hesfm_mapper_nodelet.cpp`](src/nodelets/hesfm_mapper_nodelet.cpp) update the HESFM map from incoming semantic clouds.

## Troubleshooting

### YAML changes do not seem to apply

Check whether the same parameter is also set directly in a launch file. A launch `<param ...>` entry overrides a YAML-loaded value with the same final parameter name.

### Mapper should use YAML only

The mapper and semantic cloud already load [`config/hesfm_params.yaml`](config/hesfm_params.yaml). If you want a parameter to be YAML-driven, remove or avoid duplicate launch-time overrides for that same parameter.

### Split pipeline across terminals

Use [`scripts/tmux_hesfm_pipeline.sh`](scripts/tmux_hesfm_pipeline.sh) or launch:

1. `d455_camera.launch`
2. `segmentation_only.launch launch_camera:=false`
3. `mapper_only.launch`
4. RViz separately if needed

## Citation

If you use HESFM in research, please cite:

```bibtex
@article{nguyen2026hesfm,
  title={HESFM: Hierarchical Evidential Semantic-Functional Mapping for Uncertainty-Aware Robot Navigation},
  author={Nguyen Canh, Thanh and Zhang, Haolan and HoangVan, Xiem and Chong, Nak Young},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  pages={1-8}
}
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

## Contact

- Thanh Nguyen Canh: thanhnc@jaist.ac.jp
- Prof. Nak Young Chong: chong@jaist.ac.jp
