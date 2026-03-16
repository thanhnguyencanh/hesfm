# HESFM: Hierarchical Evidential Semantic-Functional Mapping

[![ROS](https://img.shields.io/badge/ROS-Noetic-blue)](http://wiki.ros.org/noetic)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HESFM** is a novel Bayesian multi-class semantic mapping framework for robot navigation that integrates multi-source uncertainty decomposition, hierarchical Gaussian primitives, adaptive kernel inference, and information-theoretic exploration.

## Features

- **Multi-Source Uncertainty Decomposition**: Separates uncertainty into semantic (EDL), spatial consistency, observation model, and temporal components
- **Hierarchical Gaussian Primitives**: Compact scene representation with Dempster-Shafer Theory fusion
- **Adaptive Anisotropic Kernel BKI**: Uncertainty-gated, reachability-aware kernel inference
- **Extended Mutual Information Exploration**: Joint geometric and semantic information gain
- **Dual-Hardware Support**: RTX 4090/4080 development + Jetson Orin AGX deployment

## Installation

```bash
# Create workspace
mkdir -p ~/hesfm_ws/src && cd ~/hesfm_ws/src
git clone https://github.com/your-org/hesfm.git

# Install dependencies
sudo apt install ros-noetic-tf2-ros ros-noetic-pcl-ros ros-noetic-cv-bridge \
    ros-noetic-realsense2-camera ros-noetic-move-base

# Build
cd ~/hesfm_ws && catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Quick Start

```bash
# Full system with RViz
roslaunch hesfm hesfm.launch rviz:=true

# Autonomous exploration
roslaunch hesfm hesfm_exploration.launch auto_navigate:=true

# On Jetson Orin
roslaunch hesfm hesfm_jetson.launch
```

## Architecture

```
RGB-D → Segmentation → Uncertainty → Primitives → BKI Map → Costmap
                ↓           ↓           ↓           ↓
           DFormer/     4-Source    Gaussian    Adaptive
           ESANet      Decomp.     + DST       Kernel
```

## Performance

| Platform | Segmentation | Map Update | Full Pipeline |
|----------|--------------|------------|---------------|
| RTX 4090 | ~80 FPS | ~25 Hz | ~20 Hz |
| Jetson Orin | ~37 FPS | ~12 Hz | ~10 Hz |

## Citation

```bibtex
@article{nguyen2026hesfm,
  title={HESFM: Hierarchical Evidential Semantic-Functional Mapping},
  author={Nguyen, Thanh Canh and Zhang, Haolan and HoangVan, Xiem and Chong, Nak Young},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Authors

Thanh Nguyen Canh, Haolan Zhang, Xiem HoangVan, Nak Young Chong (JAIST)
