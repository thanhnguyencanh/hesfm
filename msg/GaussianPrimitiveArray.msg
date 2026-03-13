# =============================================================================
# GaussianPrimitiveArray.msg
# =============================================================================
# A collection of Gaussian primitives from a single frame
# Output of the hierarchical aggregation stage
# =============================================================================

# Header with timestamp and frame_id
std_msgs/Header header

# Array of Gaussian primitives
hesfm/GaussianPrimitive[] primitives

# -----------------------------------------------------------------------------
# Aggregation Statistics
# -----------------------------------------------------------------------------

# Number of primitives
uint32 num_primitives

# Total points aggregated
uint32 total_points

# Maximum covariance trace (for length scale normalization)
float64 max_covariance_trace

# Mean conflict across all primitives
float32 mean_conflict

# Number of primitives filtered due to high conflict
uint32 num_filtered

# Processing time for primitive construction (ms)
float32 processing_time_ms
