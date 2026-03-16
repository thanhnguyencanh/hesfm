# HESFM API Reference

## C++ API

### Namespace: hesfm

All HESFM classes are in the `hesfm` namespace.

---

### UncertaintyDecomposer

Multi-source uncertainty decomposition.

```cpp
#include "hesfm/uncertainty.h"

hesfm::UncertaintyConfig config;
config.w_semantic = 0.4;
config.w_spatial = 0.2;
config.w_observation = 0.25;
config.w_temporal = 0.15;

hesfm::UncertaintyDecomposer decomposer(config);

// Process batch of points
std::vector<SemanticPoint> points;
Eigen::Vector3d sensor_origin(0, 0, 0);
decomposer.processBatch(points, sensor_origin);
```

#### Methods

| Method | Description |
|--------|-------------|
| `computeSemanticUncertainty(probs)` | EDL-based semantic uncertainty |
| `computeSpatialUncertainty(point, neighbors)` | Spatial consistency |
| `computeObservationUncertainty(range, density, angle, model)` | Sensor model |
| `processBatch(points, sensor_origin)` | Process all points |

---

### GaussianPrimitiveBuilder

Hierarchical Gaussian primitive construction with DST fusion.

```cpp
#include "hesfm/gaussian_primitives.h"

hesfm::PrimitiveConfig config;
config.target_primitives = 128;
config.min_points_per_primitive = 5;
config.conflict_threshold = 0.3;

hesfm::GaussianPrimitiveBuilder builder(config);

std::vector<SemanticPoint> points;
auto primitives = builder.buildPrimitives(points);
```

#### Methods

| Method | Description |
|--------|-------------|
| `buildPrimitives(points)` | Build primitives from points |
| `updatePrimitives(primitives, new_points)` | Incremental update |
| `mergeSimilarPrimitives(primitives, threshold)` | Merge nearby primitives |

---

### AdaptiveKernel

Uncertainty-gated adaptive anisotropic kernels.

```cpp
#include "hesfm/adaptive_kernel.h"

hesfm::KernelConfig config;
config.length_scale_min = 0.1;
config.length_scale_max = 0.5;
config.uncertainty_threshold = 0.7;

hesfm::AdaptiveKernel kernel(config);

// Compute kernel value
double k = kernel.compute(query, primitive);

// Compute kernel matrix
Eigen::MatrixXd K = kernel.computeMatrix(queries, primitives);
```

#### Kernel Types

| Type | Description |
|------|-------------|
| `sparseKernel` | Wendland C2 compact support |
| `rbfKernel` | Gaussian RBF |
| `geometricKernel` | Mahalanobis distance |
| `uncertaintyKernel` | Uncertainty gating |
| `reachabilityKernel` | Traversability-aware |

---

### SemanticMap

Log-odds Bayesian semantic map.

```cpp
#include "hesfm/semantic_map.h"

hesfm::MapConfig config;
config.resolution = 0.05;
config.size_x = 20.0;
config.size_y = 20.0;
config.num_classes = 40;

hesfm::SemanticMap map(config);

// Update with observations
map.update(primitives, kernel);

// Query
auto cell = map.query(position);
int predicted_class = cell.state.getPredictedClass();
double confidence = cell.state.getConfidence();
```

#### Methods

| Method | Description |
|--------|-------------|
| `update(primitives, kernel)` | BKI map update |
| `query(position)` | Query cell at position |
| `getCostmap(width, height)` | Generate 2D costmap |
| `save(filename)` | Save map to file |
| `load(filename)` | Load map from file |

---

### ExplorationPlanner

EMI-based exploration goal generation.

```cpp
#include "hesfm/exploration.h"

hesfm::ExplorationConfig config;
config.max_distance = 10.0;
config.min_info_gain = 0.1;
config.sensor_range = 6.0;

hesfm::ExplorationPlanner planner(config);

// Detect frontiers
auto frontiers = planner.detectFrontiers(map);

// Compute EMI goals
auto goals = planner.computeGoals(map, robot_pose, frontiers);
```

---

### HESFMPipeline

Convenience class for full pipeline.

```cpp
#include "hesfm/hesfm.h"

hesfm::HESFMConfig config;
// ... configure all sub-components

hesfm::HESFMPipeline pipeline(config);

// Process frame
std::vector<SemanticPoint> points;
Eigen::Vector3d sensor_origin;
pipeline.process(points, sensor_origin);

// Get outputs
auto& map = pipeline.getMap();
auto costmap = pipeline.getCostmap(width, height);
```

---

## ROS API

### Messages

#### SemanticPoint.msg
```
float32 x
float32 y
float32 z
uint32 semantic_class
float32 confidence
float32 uncertainty
float32[] class_probabilities
```

#### GaussianPrimitive.msg
```
geometry_msgs/Point centroid
float64[9] covariance
uint32 semantic_class
float32 uncertainty
float32 dst_conflict
uint32 point_count
```

### Services

#### GetSemanticMap.srv
```
---
hesfm/SemanticMap map
bool success
```

#### QueryPoint.srv
```
geometry_msgs/Point point
---
uint32 semantic_class
float32 confidence
float32 uncertainty
float32[] class_probabilities
bool success
```

---

## Python API

### SemanticSegmentationNode

```python
from hesfm.scripts.semantic_segmentation_node import SemanticSegmentationNode

node = SemanticSegmentationNode()
node.run()
```

### EvaluationNode

```python
from hesfm.scripts.evaluation_node import EvaluationNode, ConfusionMatrix

# Compute metrics
cm = ConfusionMatrix(num_classes=40)
cm.update(predictions, ground_truth)
miou = cm.get_miou()
```

### CalibrationEvaluator

```python
from hesfm.scripts.evaluation_node import CalibrationEvaluator

cal = CalibrationEvaluator(num_bins=10)
cal.update(probabilities, labels)
ece = cal.get_ece()
```
