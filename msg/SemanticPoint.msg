# =============================================================================
# SemanticPoint.msg
# =============================================================================
# A single 3D point with semantic label and multi-source uncertainty
# Used for point-level semantic information before aggregation
# =============================================================================

# 3D position in the sensor/map frame
geometry_msgs/Point position

# Semantic class label (0 to num_classes-1, typically 0-39 for NYUv2)
uint8 semantic_class

# Full class probability distribution
# Length should equal num_classes (e.g., 40 for NYUv2)
float32[] class_probabilities

# -----------------------------------------------------------------------------
# Multi-source Uncertainty Decomposition
# Each component in range [0, 1], where 0 = certain, 1 = maximum uncertainty
# -----------------------------------------------------------------------------

# Semantic uncertainty from Evidential Deep Learning
# U_sem = K / S, where S = sum(alpha), alpha = evidence + 1
float32 uncertainty_semantic

# Spatial uncertainty from local neighborhood consistency
# U_spa = 1 - (same_class_neighbors / total_neighbors)
float32 uncertainty_spatial

# Observation uncertainty from sensor model
# U_obs = f(range, density, incidence_angle)
float32 uncertainty_observation

# Temporal uncertainty from prediction history
# U_temp = 1 - max_c(count_c) / total_observations
float32 uncertainty_temporal

# Total weighted uncertainty
# U_total = w_sem*U_sem + w_spa*U_spa + w_obs*U_obs + w_temp*U_temp
float32 uncertainty_total

# -----------------------------------------------------------------------------
# Additional attributes
# -----------------------------------------------------------------------------

# Depth value from sensor (meters)
float32 depth

# RGB color (for visualization)
uint8 r
uint8 g
uint8 b

# Surface normal (if available)
geometry_msgs/Vector3 normal

# Is this point on a traversable surface?
bool is_traversable
