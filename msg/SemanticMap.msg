# =============================================================================
# SemanticMap.msg
# =============================================================================
# Complete semantic map with all occupied cells
# Used for map saving/loading and full map queries
# =============================================================================

# Header with timestamp and frame_id
std_msgs/Header header

# -----------------------------------------------------------------------------
# Map Configuration
# -----------------------------------------------------------------------------

# Voxel resolution in meters
float64 resolution

# Map origin (minimum corner)
geometry_msgs/Point origin

# Map dimensions in meters
geometry_msgs/Vector3 dimensions

# Number of cells in each dimension
uint32 size_x
uint32 size_y
uint32 size_z

# Number of semantic classes
uint8 num_classes

# Class names
string[] class_names

# Traversable class indices
uint8[] traversable_classes

# -----------------------------------------------------------------------------
# Map Data
# -----------------------------------------------------------------------------

# All occupied cells
hesfm/SemanticMapCell[] cells

# Number of occupied cells
uint32 num_occupied_cells

# -----------------------------------------------------------------------------
# Map Statistics
# -----------------------------------------------------------------------------

# Map coverage (fraction of volume with observations)
float32 coverage

# Mean uncertainty across all cells
float32 mean_uncertainty

# Mean confidence across all cells
float32 mean_confidence

# Class distribution (count per class)
uint32[] class_counts

# Total number of observations integrated
uint64 total_observations

# -----------------------------------------------------------------------------
# Timing Information
# -----------------------------------------------------------------------------

# Map creation time
float64 creation_time

# Last update time
float64 last_update_time

# Total processing time (ms)
float32 total_processing_time_ms
