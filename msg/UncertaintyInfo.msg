# =============================================================================
# UncertaintyInfo.msg
# =============================================================================
# Detailed uncertainty statistics for debugging and analysis
# Published periodically to monitor system performance
# =============================================================================

# Header with timestamp
std_msgs/Header header

# -----------------------------------------------------------------------------
# Processing Statistics
# -----------------------------------------------------------------------------

# Number of points processed in this frame
uint32 num_points

# Number of points filtered due to high uncertainty
uint32 num_filtered

# Filtering threshold used
float32 filter_threshold

# -----------------------------------------------------------------------------
# Uncertainty Weights (current configuration)
# -----------------------------------------------------------------------------

float32 weight_semantic
float32 weight_spatial
float32 weight_observation
float32 weight_temporal

# -----------------------------------------------------------------------------
# Semantic Uncertainty Statistics
# From EDL: U_sem = K / S
# -----------------------------------------------------------------------------

float32 semantic_min
float32 semantic_max
float32 semantic_mean
float32 semantic_std
float32 semantic_median

# -----------------------------------------------------------------------------
# Spatial Uncertainty Statistics
# From local consistency: U_spa = 1 - consistency
# -----------------------------------------------------------------------------

float32 spatial_min
float32 spatial_max
float32 spatial_mean
float32 spatial_std
float32 spatial_median

# -----------------------------------------------------------------------------
# Observation Uncertainty Statistics
# From sensor model: U_obs = f(range, density, angle)
# -----------------------------------------------------------------------------

float32 observation_min
float32 observation_max
float32 observation_mean
float32 observation_std
float32 observation_median

# -----------------------------------------------------------------------------
# Temporal Uncertainty Statistics
# From prediction history: U_temp = 1 - max_consistency
# -----------------------------------------------------------------------------

float32 temporal_min
float32 temporal_max
float32 temporal_mean
float32 temporal_std
float32 temporal_median

# -----------------------------------------------------------------------------
# Total Uncertainty Statistics
# Weighted combination of all sources
# -----------------------------------------------------------------------------

float32 total_min
float32 total_max
float32 total_mean
float32 total_std
float32 total_median

# -----------------------------------------------------------------------------
# Histogram Data
# Distribution of total uncertainty in 10 bins [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
# -----------------------------------------------------------------------------

uint32[10] uncertainty_histogram

# Percentiles of total uncertainty
float32 percentile_25
float32 percentile_50
float32 percentile_75
float32 percentile_90
float32 percentile_95

# -----------------------------------------------------------------------------
# Correlation Analysis (optional)
# -----------------------------------------------------------------------------

# Correlation between uncertainty components
float32 corr_semantic_spatial
float32 corr_semantic_observation
float32 corr_spatial_observation
