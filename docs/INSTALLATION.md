# HESFM Installation Guide

## System Requirements

### Development Platform (High Accuracy)
- **GPU**: NVIDIA RTX 4090 or RTX 4080 (16GB+ VRAM)
- **OS**: Ubuntu 20.04 LTS
- **ROS**: Noetic
- **CUDA**: 11.8+
- **RAM**: 32GB recommended

### Deployment Platform (Real-Time)
- **Device**: NVIDIA Jetson Orin AGX
- **JetPack**: 5.1+
- **TensorRT**: 8.5+
- **Target**: >10 Hz semantic mapping

### Sensor
- Intel RealSense D455 (primary)
- Any RGB-D camera with ROS driver

## Prerequisites

### 1. ROS Noetic

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full
```

### 2. ROS Dependencies

```bash
sudo apt install \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-eigen \
    ros-noetic-pcl-ros \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-nodelet \
    ros-noetic-move-base \
    ros-noetic-navfn \
    ros-noetic-dwa-local-planner \
    ros-noetic-realsense2-camera \
    ros-noetic-visualization-msgs
```

### 3. C++ Dependencies

```bash
sudo apt install \
    libeigen3-dev \
    libpcl-dev \
    libopencv-dev \
    libyaml-cpp-dev \
    libboost-all-dev
```

### 4. Python Dependencies

```bash
pip3 install \
    torch \
    torchvision \
    numpy \
    opencv-python \
    scipy \
    sklearn \
    pyyaml
```

### 5. CUDA (for GPU inference)

```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

## Building HESFM

### Standard Build

```bash
# Create workspace
mkdir -p ~/hesfm_ws/src
cd ~/hesfm_ws/src

# Clone repository
git clone https://github.com/your-org/hesfm.git

# Build
cd ~/hesfm_ws
catkin_make -DCMAKE_BUILD_TYPE=Release

# Source
echo "source ~/hesfm_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### With OpenMP (Recommended)

```bash
catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fopenmp"
```

### Debug Build

```bash
catkin_make -DCMAKE_BUILD_TYPE=Debug
```

## Model Setup

### Download Pre-trained Models

```bash
cd ~/hesfm_ws/src/hesfm/models

# DFormerv2-Large (for RTX 4090/4080)
wget https://your-server/models/dformer_large_nyuv2.pth

# ESANet-R34-NBt1D (for Jetson or lower GPUs)
wget https://your-server/models/esanet_r34_nyuv2.pth
```

### Build TensorRT Engine (Jetson Orin)

```bash
# On Jetson device
cd ~/hesfm_ws/src/hesfm

# Convert ONNX to TensorRT
python3 scripts/build_tensorrt_engine.py \
    --onnx models/esanet_r34_nyuv2.onnx \
    --output models/esanet_r34_nyuv2_fp16.engine \
    --precision fp16
```

## Jetson Orin Setup

### JetPack Installation

1. Download JetPack 5.1+ from NVIDIA
2. Flash using SDK Manager
3. Install additional packages:

```bash
sudo apt install \
    nvidia-jetpack \
    python3-pip \
    ros-noetic-desktop
```

### Performance Optimization

```bash
# Set power mode (50W recommended)
sudo nvpmodel -m 2

# Enable max clocks
sudo jetson_clocks

# Increase swap (optional)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Intel RealSense D455 Setup

### Driver Installation

```bash
# Add repository
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"

# Install
sudo apt install librealsense2-dkms librealsense2-utils

# ROS wrapper
sudo apt install ros-noetic-realsense2-camera
```

### Verify Camera

```bash
# Check connection
realsense-viewer

# Test ROS driver
roslaunch realsense2_camera rs_camera.launch
```

## Verification

### Run Tests

```bash
cd ~/hesfm_ws
catkin_make run_tests

# Or specific tests
rostest hesfm test_uncertainty.test
rostest hesfm test_primitives.test
```

### Launch System

```bash
# Basic launch
roslaunch hesfm hesfm.launch rviz:=true

# Check topics
rostopic list | grep hesfm
rostopic hz /semantic_map
```

## Troubleshooting

### CUDA Not Found
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### TensorRT Import Error
```bash
pip3 install tensorrt
# Or on Jetson: already included in JetPack
```

### PCL Compilation Issues
```bash
sudo apt install libpcl-dev
# Ensure PCL version >= 1.10
```

### RealSense Permission Denied
```bash
sudo cp ~/.../99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```
