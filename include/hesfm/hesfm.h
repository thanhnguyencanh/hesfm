/**
 * @file hesfm.h
 * @brief Main header file for HESFM (Hierarchical Evidential Semantic-Functional Mapping)
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * @mainpage HESFM: Hierarchical Evidential Semantic-Functional Mapping
 */

#ifndef HESFM_HESFM_H_
#define HESFM_HESFM_H_

// Version information
#define HESFM_VERSION_MAJOR 1
#define HESFM_VERSION_MINOR 0
#define HESFM_VERSION_PATCH 0
#define HESFM_VERSION_STRING "1.0.0"

// Core headers
#include "hesfm/types.h"
#include "hesfm/config.h"

// Processing modules
#include "hesfm/uncertainty.h"
#include "hesfm/gaussian_primitives.h"
#include "hesfm/adaptive_kernel.h"

// Map representation
#include "hesfm/semantic_map.h"

// Exploration
#include "hesfm/exploration.h"

namespace hesfm {

/**
 * @brief Get HESFM version string
 */
inline std::string getVersionString() {
    return HESFM_VERSION_STRING;
}

/**
 * @brief Get HESFM version as tuple
 */
inline std::tuple<int, int, int> getVersion() {
    return {HESFM_VERSION_MAJOR, HESFM_VERSION_MINOR, HESFM_VERSION_PATCH};
}

/**
 * @brief HESFM Pipeline - Convenience wrapper for full processing pipeline
 * 
 * Encapsulates all processing stages in a single class for ease of use.
 * For more control, use individual components directly.
 */
class HESFMPipeline {
public:
    /**
     * @brief Constructor with configuration
     */
    explicit HESFMPipeline(const HESFMConfig& config = HESFMConfig())
        : config_(config),
          uncertainty_decomposer_(config.uncertainty),
          primitive_builder_(config.primitive),
          kernel_(config.kernel),
          semantic_map_(config.map),
          exploration_planner_(config.exploration) {}
    
    /**
     * @brief Process semantic point cloud and update map
     * 
     * Full pipeline:
     * 1. Uncertainty decomposition
     * 2. Gaussian primitive construction
     * 3. Map update via BKI
     * 
     * @param points Semantic points from segmentation
     * @param sensor_origin Sensor position
     * @return Number of primitives generated
     */
    int process(std::vector<SemanticPoint>& points, const Vector3d& sensor_origin) {
        // Step 1: Compute uncertainties
        uncertainty_decomposer_.processPointCloud(points, sensor_origin);

        // Step 2: Build primitives (cache result — avoids double K-means in node)
        last_primitives_ = primitive_builder_.buildPrimitives(points);

        // Step 3: Update map
        semantic_map_.update(last_primitives_, kernel_);

        return static_cast<int>(last_primitives_.size());
    }

    /**
     * @brief Return primitives from the most recent process() call (no recompute)
     */
    const std::vector<GaussianPrimitive>& getLastPrimitives() const {
        return last_primitives_;
    }
    
    /**
     * @brief Get exploration goals
     */
    std::vector<ExplorationGoal> getExplorationGoals(
        const Vector3d& robot_position,
        const Quaterniond& robot_orientation = Quaterniond::Identity()) {
        return exploration_planner_.computeGoals(semantic_map_, robot_position, robot_orientation);
    }
    
    /**
     * @brief Generate 2D costmap for navigation
     */
    std::vector<int8_t> getCostmap(int& width, int& height) {
        return semantic_map_.generateCostmap(
            config_.navigation.costmap_height_min,
            config_.navigation.costmap_height_max,
            width, height);
    }
    
    /**
     * @brief Query semantic class at position
     */
    int getClass(const Vector3d& position) const {
        return semantic_map_.getClass(position);
    }
    
    /**
     * @brief Query if position is traversable
     */
    bool isTraversable(const Vector3d& position) const {
        return semantic_map_.isTraversable(position);
    }
    
    /**
     * @brief Reset map
     */
    void reset() {
        semantic_map_.reset();
        uncertainty_decomposer_.resetTemporalHistory();
    }
    
    // Accessors for individual components
    UncertaintyDecomposer& getUncertaintyDecomposer() { return uncertainty_decomposer_; }
    GaussianPrimitiveBuilder& getPrimitiveBuilder() { return primitive_builder_; }
    AdaptiveKernel& getKernel() { return kernel_; }
    SemanticMap& getMap() { return semantic_map_; }
    ExplorationPlanner& getExplorationPlanner() { return exploration_planner_; }
    
    const HESFMConfig& getConfig() const { return config_; }

private:
    HESFMConfig config_;
    UncertaintyDecomposer uncertainty_decomposer_;
    GaussianPrimitiveBuilder primitive_builder_;
    AdaptiveKernel kernel_;
    SemanticMap semantic_map_;
    ExplorationPlanner exploration_planner_;
    std::vector<GaussianPrimitive> last_primitives_;
};

} // namespace hesfm

#endif // HESFM_HESFM_H_
