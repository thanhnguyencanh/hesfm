# =============================================================================
# SemanticMapCell.msg
# =============================================================================
# A single cell in the semantic map with extended state
# Contains log-odds representation and additional navigation info
# =============================================================================

# Cell position (center of voxel)
geometry_msgs/Point position

# Grid indices
int32 index_x
int32 index_y
int32 index_z

# -----------------------------------------------------------------------------
# Semantic State (Log-Odds Representation)
# -----------------------------------------------------------------------------

# Log-odds values for each class
# P(c) = exp(h_c) / sum(exp(h))
float32[] log_odds

# Predicted semantic class (argmax of probabilities)
uint8 semantic_class

# Class probabilities (derived from log-odds via softmax)
float32[] class_probabilities

# Maximum probability (confidence)
float32 confidence

# -----------------------------------------------------------------------------
# Observation Information
# -----------------------------------------------------------------------------

# Number of observations integrated into this cell
uint32 observation_count

# Time of last update (seconds since epoch)
float64 last_update_time

# Time since last observation (for temporal decay)
float32 time_since_observation

# -----------------------------------------------------------------------------
# Extended State for Navigation
# -----------------------------------------------------------------------------

# Is this cell occupied by an obstacle?
bool is_occupied

# Is this cell traversable (floor, floor_mat)?
bool is_traversable

# Is this cell associated with a dynamic object?
bool is_dynamic

# Reachability probability [0, 1]
# High for floor connected to robot, low for isolated regions
float32 reachability

# Navigation cost [0, 100]
# 0 = free, 100 = lethal obstacle, -1 = unknown
int8 nav_cost

# -----------------------------------------------------------------------------
# Uncertainty
# -----------------------------------------------------------------------------

# Aggregated uncertainty for this cell
float32 uncertainty

# Entropy of class distribution
# H = -sum(p * log(p))
float32 entropy
