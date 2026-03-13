# =============================================================================
# ExplorationGoal.msg
# =============================================================================
# Exploration goal based on Extended Mutual Information (EMI)
# Used for uncertainty-aware autonomous exploration
# =============================================================================

# Header with timestamp and frame_id
std_msgs/Header header

# -----------------------------------------------------------------------------
# Goal Position
# -----------------------------------------------------------------------------

# Target position for exploration
geometry_msgs/Point position

# Target orientation (viewing direction)
geometry_msgs/Quaternion orientation

# Goal as full pose
geometry_msgs/Pose pose

# -----------------------------------------------------------------------------
# Information Metrics
# -----------------------------------------------------------------------------

# Extended Mutual Information value
# EMI = sum over cells of: P(occupied) * H(semantic | observation)
float64 emi_value

# Expected information gain
float64 expected_info_gain

# Uncertainty reduction potential
float64 uncertainty_reduction

# Coverage improvement potential
float32 coverage_improvement

# -----------------------------------------------------------------------------
# Goal Properties
# -----------------------------------------------------------------------------

# Frontier ID (if this goal is on a frontier)
uint32 frontier_id

# Number of unknown cells visible from this goal
uint32 num_unknown_visible

# Number of high-uncertainty cells visible
uint32 num_uncertain_visible

# Distance from current robot position
float32 distance_to_goal

# Estimated travel time (seconds)
float32 estimated_travel_time

# Path safety score [0, 1]
float32 path_safety

# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------

# Overall utility score (combination of all factors)
float64 utility_score

# Rank among all candidate goals (1 = best)
uint32 rank

# -----------------------------------------------------------------------------
# Validity
# -----------------------------------------------------------------------------

# Is this goal reachable?
bool is_reachable

# Is this goal still valid? (may become invalid as map updates)
bool is_valid

# Expiration time (goal should be re-evaluated after this)
float64 expiration_time