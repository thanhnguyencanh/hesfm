# =============================================================================
# SemanticPointCloud.msg
# =============================================================================
# A collection of semantic points with metadata
# Represents a single frame of semantic point cloud data
# =============================================================================

# Header with timestamp and frame_id
std_msgs/Header header

# Array of semantic points
hesfm/SemanticPoint[] points

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------

# Number of semantic classes
uint8 num_classes

# Class names (optional, for debugging/visualization)
string[] class_names

# Sensor origin at time of capture
geometry_msgs/Point sensor_origin

# Sensor orientation at time of capture
geometry_msgs/Quaternion sensor_orientation

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

# Total number of points
uint32 num_points

# Number of valid points (after filtering)
uint32 num_valid_points

# Mean uncertainty
float32 mean_uncertainty

# Processing time in milliseconds
float32 processing_time_ms
