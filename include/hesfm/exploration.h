/**
 * @file exploration.h
 * @brief Extended Mutual Information based exploration for HESFM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * Implements EMI-guided exploration for uncertainty-aware autonomous navigation.
 * EMI considers both geometric (occupancy) and semantic uncertainty:
 * 
 * EMI(x) = Σ_c P(occupied_c | x) · H(semantic | z, x)
 * 
 * Where:
 * - P(occupied_c | x): Probability that cell c is observed from pose x
 * - H(semantic | z, x): Conditional entropy of semantic prediction
 */

#ifndef HESFM_EXPLORATION_H_
#define HESFM_EXPLORATION_H_

#include "hesfm/types.h"
#include "hesfm/config.h"
#include "hesfm/semantic_map.h"
#include <memory>

namespace hesfm {

/**
 * @brief Frontier cell information
 */
struct FrontierCell {
    Vector3d position;
    int frontier_id;
    double information_gain;
    int num_unknown_neighbors;
};

/**
 * @brief Frontier region information
 */
struct Frontier {
    int id;
    std::vector<FrontierCell> cells;
    Vector3d centroid;
    double size;  // Number of cells
    double total_info_gain;
    double mean_info_gain;
};

/**
 * @brief EMI-based exploration planner
 * 
 * Computes exploration goals based on information gain that considers
 * both unknown space and semantic uncertainty.
 * 
 * @code
 * ExplorationConfig config;
 * ExplorationPlanner planner(config);
 * 
 * SemanticMap map;
 * geometry_msgs::Pose robot_pose;
 * auto goals = planner.computeGoals(map, robot_pose);
 * @endcode
 */
class ExplorationPlanner {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    ExplorationPlanner();
    explicit ExplorationPlanner(const ExplorationConfig& config);
    ~ExplorationPlanner() = default;
    
    // =========================================================================
    // Main Interface
    // =========================================================================
    
    /**
     * @brief Compute exploration goals
     * 
     * @param map Current semantic map
     * @param robot_position Current robot position
     * @param robot_orientation Current robot orientation
     * @return Sorted vector of exploration goals (best first)
     */
    std::vector<ExplorationGoal> computeGoals(
        const SemanticMap& map,
        const Vector3d& robot_position,
        const Quaterniond& robot_orientation = Quaterniond::Identity());
    
    /**
     * @brief Get best exploration goal
     */
    std::optional<ExplorationGoal> getBestGoal(
        const SemanticMap& map,
        const Vector3d& robot_position,
        const Quaterniond& robot_orientation = Quaterniond::Identity());
    
    /**
     * @brief Check if exploration is complete
     */
    bool isExplorationComplete(const SemanticMap& map) const;
    
    // =========================================================================
    // Frontier Detection
    // =========================================================================
    
    /**
     * @brief Detect frontiers in the map
     * 
     * Frontiers are boundaries between known and unknown space
     */
    std::vector<Frontier> detectFrontiers(const SemanticMap& map) const;
    
    /**
     * @brief Get frontier cells
     */
    std::vector<FrontierCell> getFrontierCells(const SemanticMap& map) const;
    
    /**
     * @brief Cluster frontier cells into regions
     */
    std::vector<Frontier> clusterFrontiers(
        const std::vector<FrontierCell>& cells) const;
    
    // =========================================================================
    // Information Gain Computation
    // =========================================================================
    
    /**
     * @brief Compute Extended Mutual Information for a viewpoint
     * 
     * EMI(x) = Σ_c P(visible_c | x) · [H_geo(c) + λ · H_sem(c)]
     * 
     * @param map Semantic map
     * @param viewpoint Candidate viewpoint position
     * @param orientation Viewing direction
     * @return EMI value
     */
    double computeEMI(const SemanticMap& map,
                       const Vector3d& viewpoint,
                       const Quaterniond& orientation) const;
    
    /**
     * @brief Compute geometric information gain (occupancy entropy)
     */
    double computeGeometricInfoGain(const SemanticMap& map,
                                     const Vector3d& viewpoint,
                                     const Quaterniond& orientation) const;
    
    /**
     * @brief Compute semantic information gain (semantic entropy)
     */
    double computeSemanticInfoGain(const SemanticMap& map,
                                    const Vector3d& viewpoint,
                                    const Quaterniond& orientation) const;
    
    /**
     * @brief Compute expected uncertainty reduction
     */
    double computeUncertaintyReduction(const SemanticMap& map,
                                        const Vector3d& viewpoint,
                                        const Quaterniond& orientation) const;
    
    // =========================================================================
    // Visibility Computation
    // =========================================================================
    
    /**
     * @brief Get cells visible from a viewpoint
     */
    std::vector<Vector3d> getVisibleCells(const SemanticMap& map,
                                           const Vector3d& viewpoint,
                                           const Quaterniond& orientation) const;
    
    /**
     * @brief Check if a cell is visible from viewpoint
     */
    bool isCellVisible(const SemanticMap& map,
                        const Vector3d& cell_position,
                        const Vector3d& viewpoint,
                        const Quaterniond& orientation) const;
    
    /**
     * @brief Raycast to check visibility
     */
    bool raycast(const SemanticMap& map,
                  const Vector3d& start,
                  const Vector3d& end) const;
    
    // =========================================================================
    // Goal Evaluation
    // =========================================================================
    
    /**
     * @brief Compute utility score for a goal
     * 
     * Combines information gain, distance, and safety
     */
    double computeUtility(const ExplorationGoal& goal,
                           const Vector3d& robot_position) const;
    
    /**
     * @brief Check if goal is reachable
     */
    bool isGoalReachable(const SemanticMap& map,
                          const Vector3d& goal,
                          const Vector3d& robot_position) const;
    
    /**
     * @brief Check if goal is safe (sufficient distance from obstacles)
     */
    bool isGoalSafe(const SemanticMap& map, const Vector3d& goal) const;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    const ExplorationConfig& getConfig() const { return config_; }
    void setConfig(const ExplorationConfig& config) { config_ = config; }

private:
    ExplorationConfig config_;
    
    /**
     * @brief Sample candidate viewpoints around frontiers
     */
    std::vector<std::pair<Vector3d, Quaterniond>> sampleViewpoints(
        const std::vector<Frontier>& frontiers,
        const Vector3d& robot_position) const;
    
    /**
     * @brief Compute viewing direction towards target
     */
    Quaterniond computeViewingDirection(const Vector3d& from,
                                          const Vector3d& target) const;
    
    /**
     * @brief Check if position is within sensor FOV
     */
    bool isInFOV(const Vector3d& point,
                  const Vector3d& viewpoint,
                  const Quaterniond& orientation) const;
};

} // namespace hesfm

#endif // HESFM_EXPLORATION_H_
