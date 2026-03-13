# =============================================================================
# ExplorationGoalArray.msg
# =============================================================================
# Array of ranked exploration goals
# =============================================================================

# Header with timestamp and frame_id
std_msgs/Header header

# Array of exploration goals (sorted by utility, best first)
hesfm/ExplorationGoal[] goals

# Number of goals
uint32 num_goals

# Best goal (convenience accessor)
hesfm/ExplorationGoal best_goal

# Current map coverage [0, 1]
float32 current_coverage

# Current mean uncertainty
float32 current_mean_uncertainty

# Is exploration complete? (coverage > threshold and uncertainty < threshold)
bool exploration_complete

# Processing time for goal generation (ms)
float32 processing_time_ms
