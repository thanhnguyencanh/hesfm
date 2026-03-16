# HESFM Usage Guide

## Basic Usage

### Launch Full System

```bash
# With D455 camera and RViz visualization
roslaunch hesfm hesfm.launch rviz:=true

# Without camera (for rosbag playback)
roslaunch hesfm hesfm.launch launch_camera:=false

# Specify segmentation backend
roslaunch hesfm hesfm.launch segmentation_backend:=dformer
roslaunch hesfm hesfm.launch segmentation_backend:=esanet_trt
```

### Jetson Orin Deployment

```bash
# Optimized for real-time on Jetson
roslaunch hesfm hesfm_jetson.launch

# With specific power mode
roslaunch hesfm hesfm_jetson.launch power_mode:=MAXN
```

## Exploration Mode

### Autonomous Exploration

```bash
# Launch with exploration
roslaunch hesfm hesfm_exploration.launch

# With automatic navigation to goals
roslaunch hesfm hesfm_exploration.launch auto_navigate:=true
```

## Navigation Integration

```bash
# Full navigation stack
roslaunch hesfm hesfm_navigation.launch
```

## Evaluation

### On NYUv2 Dataset

```bash
roslaunch hesfm hesfm_evaluation.launch \
    dataset:=nyuv2 \
    data_path:=/path/to/nyuv2 \
    split:=test
```

## Runtime Configuration

### Dynamic Reconfigure

```bash
rosrun rqt_reconfigure rqt_reconfigure

# Or command line
rosrun dynamic_reconfigure dynparam set /hesfm_mapper_node w_semantic 0.5
```

## Services

```bash
# Reset map
rosservice call /hesfm_mapper_node/reset_map

# Save map
rosservice call /hesfm_mapper_node/save_map "filename: '/tmp/map.yaml'"
```

## Topics

### Input
- `/camera/color/image_raw` - RGB image
- `/camera/aligned_depth_to_color/image_raw` - Depth image

### Output
- `/semantic_map` - 3D semantic map
- `/costmap` - 2D navigation costmap
- `/primitives` - Gaussian primitives
- `/exploration_goal` - Best exploration goal

## Troubleshooting

### No Point Cloud Output
1. Check camera: `rostopic hz /camera/color/image_raw`
2. Check TF: `rosrun tf tf_echo map camera_color_optical_frame`

### Low FPS
1. Reduce resolution: `resolution:=0.08`
2. Use Jetson config: `hesfm_jetson.launch`
