<!-- # Semantic-Octomap
Semantic 3-D OctoMap implementation for building probabilistic multi-class maps of an environment using semantic pointcloud input.

## Dependencies
* ROS Melodic or Noetic
* Catkin
* PCL
* OctoMap
* octomap_msgs
* octomap_rviz_plugins
* scikit-image

## How to use
1. `launch/semantic_octomap.launch` starts ROS nodes `semantic_cloud` and `semantic_octomap_node`.
2. `semantic_cloud` takes three **aligned** images, i.e. RBG image, depth image, and semantic segmentation image. The output is a sematic pointcloud topic of type `sensor_msgs/PointCloud2`.
3. `semantic_octomap_node` receives the generated semantic pointcloud and updates the semantic OctoMap. This node internally maintains a semantic OcTree where each node stores the probability of each object class. Two types of semantic OctoMap topic are published as instances of `octomap_msgs/Octomap` message: `octomap_full` and `octomap_color`. `octomap_full` contains the full probability distribution over the object classes, while `octomap_color` only stores the maximmum likelihood semantic OcTree (with probabilistic occupancies). A probabilistic 2-D occupancy map topic is additionally published via projection of the OctoMap on the ground plane.
4. Note that `octomap_rviz_plugins` can only visualize `octomap_color`, whereas visualizing `octomap_full` causes Rviz to crash.
5. `params/semantic_cloud.yaml` stores camera intrinsic parameters. This should be set to the values used by your camera.
6. `params/octomap_generator.yaml` stores parameters for the semantic OctoMap such as minimum grid size (`resolution`), log-odds increments (`psi` and `phi`), and path for saving the final map. -->

# HESFM: Hierarchical Evidential Semantic-Functional Mapping

[![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

A novel semantic mapping framework for uncertainty-aware robot navigation in indoor environments.

<p align="center">
  <img src="docs/images/hesfm_overview.png" alt="HESFM Overview" width="800"/>
</p>

## Key Features

- **Multi-Source Uncertainty Decomposition**: Separates uncertainty into semantic, spatial, observation, and temporal components
- **Hierarchical Gaussian Primitives**: Efficient point cloud aggregation using uncertainty-weighted clustering and Dempster-Shafer fusion
- **Adaptive Anisotropic Kernel**: Geometry-aware BKI with uncertainty gating and reachability constraints
- **Extended Semantic State**: Beyond class probabilities - includes dynamic status, reachability, and affordances
- **Real-Time Performance**: 15+ Hz on RTX 4090, 10+ Hz on Jetson Orin AGX
- **Navigation Ready**: Direct integration with ROS navigation stack via costmap output

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Nodes](#nodes)
- [Topics](#topics)
- [Parameters](#parameters)
- [Launch Files](#launch-files)
- [Citation](#citation)
- [License](#license)

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Camera** | Intel RealSense D435 | Intel RealSense D455 |
| **GPU (Development)** | NVIDIA RTX 3060 | NVIDIA RTX 4090 |
| **GPU (Deployment)** | Jetson Xavier NX | Jetson Orin AGX |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 20 GB | 50+ GB (with models) |

### Software

| Software | Version |
|----------|---------|
| Ubuntu | 20.04 LTS |
| ROS | Noetic |
| CUDA | 11.8+ |
| cuDNN | 8.6+ |
| PyTorch | 2.0+ |
| TensorRT | 8.5+ (for Jetson) |
| Python | 3.8+ |

## Installation


### 1. Create Workspace

```bash
# Create catkin workspace
mkdir -p ~/hesfm_ws/src
cd ~/hesfm_ws/src

# Clone HESFM package
git clone https://github.com/jaist-robotics/hesfm.git

# Clone semantic segmentation models (choose one)
# Option A: DFormer (high accuracy)
git clone https://github.com/VCIP-RGBD/DFormer.git

# Option B: ESANet (real-time)
git clone https://github.com/TUI-NICR/ESANet.git
```

<!-- ### 2. Python Environment

```bash
# Create conda environment
conda create -n hesfm python=3.10 -y
conda activate hesfm

# Install PyTorch (CUDA 11.8)
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional Python packages
pip install \
    scipy \
    scikit-learn \
    open3d \
    tensorboard \
    rospkg \
    catkin_pkg \
    empy==3.3.4

# For DFormer
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
``` -->
### 2. Docker
```
sudo docker build -f Dockerfile.blackwell -t activesemanticslam:latest . 2>&1 | tee build.log

sudo docker stop semanticslam_ros

sudo docker rm semanticslam_ros

sudo rm -rf ~/.bash_aliases

touch ~/.bash_aliases

sudo ./run_semantic_slam_docker.sh

docker start semanticslam_ros
docker exec -it semanticslam_ros /bin/bash
```

### 3. Prerequisites

```bash
# Install ROS Noetic (if not already installed)
sudo apt update
sudo apt install ros-noetic-desktop-full

# Install RealSense SDK and ROS packages
sudo apt install ros-noetic-realsense2-camera ros-noetic-realsense2-description

# Install additional ROS dependencies
sudo apt install \
    ros-noetic-pcl-ros \
    ros-noetic-cv-bridge \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-image-transport \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-nodelet

# Install system dependencies
sudo apt install \
    libeigen3-dev \
    libpcl-dev \
    libopencv-dev \
    libboost-all-dev
```

# Install vision_opencv 
```
git clone -b noetic https://github.com/ros-perception/vision_opencv
```

### 4. Download Pretrained Models

```bash
cd ~/hesfm_ws/src/hesfm

# Create checkpoints directory
mkdir -p checkpoints

# Download DFormer weights (NYUv2)
# Visit: https://github.com/VCIP-RGBD/DFormer#pretrained-models
# Download DFormer_Large_NYU.pth to checkpoints/

# Download ESANet weights (NYUv2)
cd checkpoints
gdown 1C5-kJv4w3foicEudP3DAjdIXVuzUK7O8
tar -xvzf nyuv2_r34_NBt1D.tar.gz
```

#### On the desktop 

Step 1: Create clean environment
```
conda create -n esanet_trt python=3.10 -y
conda activate esanet_trt
```

Step 2: Install compatible versions
```
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
pip install onnx==1.14.1 numpy==1.24.0
```

Step 3: Export
```
cd ~/sslam_ws/src/ESANet
python model_to_onnx.py \
    --dataset nyuv2 \
    --height 480 --width 640 \
    --encoder resnet34 \
    --encoder_block NonBottleneck1D \
    --modality rgbd \
    --last_ckpt ./trained_models/nyuv2/r34_NBt1D.pth \
    --model_output_name esanet_r34_nyuv2_v11 \
    --onnx_opset_version 11
```

Step 4: Verify
```
python -c "
import onnx
m = onnx.load('onnx_models/esanet_r34_nyuv2_v11.onnx')
print(f'Opset: {m.opset_import[0].version}')
onnx.checker.check_model(m)
print('✅ Valid')
"
```
#### Copy to Jetson
#### On Xavier - build TensorRT engine
```
/usr/src/tensorrt/bin/trtexec \
    --onnx=/home/jetson/models/esanet_r34_nyuv2_v11.onnx \
    --saveEngine=/home/jetson/models/esanet_r34_nyuv2_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:1024MiB
```

### 5. Build Package

```bash
cd ~/hesfm_ws

# Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build
catkin build hesfm

# Source workspace
source devel/setup.bash
echo "source ~/hesfm_ws/devel/setup.bash" >> ~/.bashrc
```

## Quick Start

### Basic Usage

```bash
# Terminal 1: Launch HESFM with D455 camera
roslaunch hesfm hesfm.launch

# Terminal 2: Visualize in RViz (if not using use_rviz:=true)
rviz -d $(rospack find hesfm)/rviz/hesfm.rviz
```

### Development Mode (High Accuracy)

```bash
# Uses DFormer on RTX GPU
roslaunch hesfm hesfm.launch backend:=dformer use_rviz:=true
```

### Deployment Mode (Real-Time)

```bash
# Uses ESANet + TensorRT on Jetson
roslaunch hesfm hesfm.launch backend:=esanet_trt use_rviz:=false
```

### With Navigation Stack

```bash
# Launch HESFM with move_base integration
roslaunch hesfm hesfm_navigation.launch
```

## Package Structure

```
hesfm/
├── CMakeLists.txt              # Build configuration
├── package.xml                 # Package manifest
├── README.md                   # This file
├── LICENSE                     # BSD-3-Clause license
│
├── include/hesfm/              # C++ headers
│   ├── hesfm.h                 # Main header 
│   ├── types.h                 # Type definitions
│   ├── uncertainty.h           # Uncertainty decomposition
│   ├── gaussian_primitives.h   # Gaussian primitive builder
│   ├── adaptive_kernel.h       # Adaptive anisotropic kernel
│   ├── semantic_map.h          # Semantic map with log-odds update
│   └── hesfm.h                 # Pipeline convenience wrapper
│
├── src/
│   ├── hesfm_core/             # Core library implementations
│   │   ├── uncertainty.cpp
│   │   ├── gaussian_primitives.cpp
│   │   ├── adaptive_kernel.cpp
│   │   └── semantic_map.cpp
│   │
│   ├── nodes/                  # ROS node implementations
│   │   ├── hesfm_mapper_node.cpp
│   │   ├── semantic_cloud_node.cpp
│   │   ├── costmap_node.cpp
│   │   └── visualization_node.cpp
│   │
│   └── nodelets/               # Nodelet implementations
│       ├── hesfm_mapper_nodelet.cpp
│       └── semantic_cloud_nodelet.cpp
│
├── scripts/                    # Python nodes
│   ├── semantic_segmentation_node.py
│   ├── dformer_node.py
│   ├── esanet_node.py
│   ├── semantic_cloud_node.py
│   └── evaluation_node.py
│
├── launch/                     # Launch files
│   ├── hesfm.launch            # Main launch file
│   ├── d455_camera.launch      # D455 camera configuration
│   ├── hesfm_navigation.launch # With navigation stack
│   └── hesfm_exploration.launch
│
├── config/                     # Configuration files
│   ├── hesfm_params.yaml       # Main runtime parameters
│   └── benchmark_config.yaml   # Offline evaluation helper config
│
├── rviz/                       # RViz configurations
│   └── hesfm.rviz
│
├── msg/                        # Custom messages
│   ├── SemanticPoint.msg
│   ├── SemanticPrimitive.msg
│   ├── SemanticMapInfo.msg
│   └── UncertaintyInfo.msg
│
├── srv/                        # Custom services
│   ├── GetSemanticMap.srv
│   ├── QuerySemanticClass.srv
│   ├── ResetMap.srv
│   └── SaveMap.srv
│
├── checkpoints/                # Model weights (download separately)
│
├── test/                       # Unit and integration tests
│
└── docs/                       # Documentation
    ├── images/
    └── api/
```

## Nodes

### hesfm_mapper_node

Main semantic mapping node that processes semantic point clouds and maintains the semantic map.

**Subscribed Topics:**
- `semantic_cloud` ([sensor_msgs/PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html)): Semantic point cloud with labels and uncertainties

**Published Topics:**
- `semantic_map` ([sensor_msgs/PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html)): 3D semantic map
- `costmap` ([nav_msgs/OccupancyGrid](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html)): 2D navigation costmap
- `primitives` ([visualization_msgs/MarkerArray](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html)): Gaussian primitive visualization

### semantic_segmentation_node.py

Python node for RGB-D semantic segmentation using DFormer or ESANet.

**Subscribed Topics:**
- `rgb` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): RGB image
- `depth` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): Depth image

**Published Topics:**
- `/hesfm/semantic_image` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): Semantic labels (mono8)
- `/hesfm/semantic_color` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): Colored visualization
- `/hesfm/semantic_probs` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): Class probabilities

## Topics

### Input Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | sensor_msgs/Image | RGB image from camera |
| `/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image | Aligned depth image |
| `/camera/color/camera_info` | sensor_msgs/CameraInfo | Camera intrinsics |

### Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/hesfm/semantic_map` | sensor_msgs/PointCloud2 | 3D semantic map |
| `/hesfm/costmap` | nav_msgs/OccupancyGrid | 2D navigation costmap |
| `/hesfm/semantic_image` | sensor_msgs/Image | Semantic segmentation labels |
| `/hesfm/semantic_color` | sensor_msgs/Image | Colored semantic visualization |
| `/hesfm/primitives` | visualization_msgs/MarkerArray | Gaussian primitives |
| `/hesfm/uncertainty_map` | sensor_msgs/PointCloud2 | Uncertainty visualization |

## Parameters

### Main Parameters (~hesfm_mapper_node)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | double | 0.05 | Voxel resolution in meters |
| `num_classes` | int | 40 | Number of semantic classes |
| `map_frame` | string | "map" | Map frame ID |
| `target_primitives` | int | 128 | Target number of Gaussian primitives |

### Uncertainty Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_semantic` | double | 0.4 | Semantic uncertainty weight |
| `w_spatial` | double | 0.2 | Spatial uncertainty weight |
| `w_observation` | double | 0.25 | Observation uncertainty weight |
| `w_temporal` | double | 0.15 | Temporal uncertainty weight |

### Kernel Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `length_scale_min` | double | 0.1 | Minimum kernel length scale |
| `length_scale_max` | double | 0.5 | Maximum kernel length scale |
| `uncertainty_threshold` | double | 0.7 | Uncertainty gating threshold |

## Launch Files

### hesfm.launch

Main launch file with all components.

```xml
<launch>
  <arg name="use_rviz" default="true"/>
  <arg name="backend" default="auto"/>  <!-- auto, dformer, esanet_trt -->
  <arg name="camera" default="camera"/>
  
  <include file="$(find hesfm)/launch/hesfm.launch">
    <arg name="use_rviz" value="$(arg use_rviz)"/>
    <arg name="backend" value="$(arg backend)"/>
  </include>
</launch>
```

## Mathematical Formulation

### Uncertainty Decomposition

$$U_{total} = w_{sem} U_{sem} + w_{spa} U_{spa} + w_{obs} U_{obs} + w_{temp} U_{temp}$$

### Gaussian Primitive

$$G_j = (\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j, \mathbf{p}_j, u_j)$$

### Adaptive Kernel

$$\tilde{k}(\mathbf{x}, G_j) = k_{geo} \cdot k_{unc} \cdot k_{reach}$$

### Log-Odds Update

$$\mathbf{h}_{t+1} = \mathbf{h}_t + \sum_j \tilde{k}(\mathbf{x}, G_j) \cdot w(U_j) \cdot \mathbf{l}^{score}(G_j)$$

## Citation

If you use HESFM in your research, please cite:

```bibtex
@article{nguyen2026hesfm,
  title={{HESFM}: Hierarchical Evidential Semantic-Functional Mapping 
         for Uncertainty-Aware Robot Navigation},
  author={Nguyen Canh, Thanh and Zhang, Haolan and HoangVan, Xiem and Chong, Nak Young},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  volume={},
  number={},
  pages={1-8},
  doi={}
}
```

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DFormer](https://github.com/VCIP-RGBD/DFormer) for RGB-D semantic segmentation
- [ESANet](https://github.com/TUI-NICR/ESANet) for efficient real-time segmentation
- [S-BKI](https://github.com/ganlumomo/BKI-ROS) for Bayesian kernel inference
- [EvSemMap](https://github.com/junwon-vision/EvSemMap) for evidential semantic mapping

## Contact

- **Thanh Nguyen Canh** - thanhnc@jaist.ac.jp
- **Prof. Nak Young Chong** - chong@jaist.ac.jp

School of Information Science, Japan Advanced Institute of Science and Technology (JAIST)
