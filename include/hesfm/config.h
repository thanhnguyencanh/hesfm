/**
 * @file config.h
 * @brief Configuration structures for HESFM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * Contains all configuration structures for the HESFM framework components.
 */

#ifndef HESFM_CONFIG_H_
#define HESFM_CONFIG_H_

#include "hesfm/types.h"
#include <string>

namespace hesfm {

// =============================================================================
// Uncertainty Configuration
// =============================================================================

/**
 * @brief Configuration for uncertainty decomposition
 */
struct UncertaintyConfig {
    /// Uncertainty weights (should sum to 1.0)
    double w_semantic = 0.4;      ///< Weight for semantic uncertainty
    double w_spatial = 0.2;       ///< Weight for spatial uncertainty
    double w_observation = 0.25;  ///< Weight for observation uncertainty
    double w_temporal = 0.15;     ///< Weight for temporal uncertainty
    
    /// Spatial uncertainty parameters
    double spatial_radius = 0.2;  ///< Neighborhood radius (meters)
    int min_neighbors = 3;        ///< Minimum neighbors for valid computation
    
    /// Observation uncertainty coefficients
    double sigma_range = 0.5;
    double sigma_density = 0.3;
    double sigma_angle = 0.2;
    double max_density = 100.0;
    
    /// Temporal tracking
    double temporal_resolution = 0.05;  ///< Spatial hash resolution
    int temporal_window = 10;           ///< Number of observations to track
    
    /**
     * @brief Validate that weights sum to 1.0
     */
    bool validateWeights() const {
        double sum = w_semantic + w_spatial + w_observation + w_temporal;
        return std::abs(sum - 1.0) < 0.001;
    }
    
    /**
     * @brief Normalize weights to sum to 1.0
     */
    void normalizeWeights() {
        double sum = w_semantic + w_spatial + w_observation + w_temporal;
        if (sum > 0) {
            w_semantic /= sum;
            w_spatial /= sum;
            w_observation /= sum;
            w_temporal /= sum;
        }
    }
};

// =============================================================================
// Gaussian Primitive Configuration
// =============================================================================

/**
 * @brief Configuration for Gaussian primitive construction
 */
struct PrimitiveConfig {
    /// Target number of primitives per frame
    int target_primitives = 128;
    
    /// Minimum points required to form a primitive
    int min_points_per_primitive = 5;
    
    /// Maximum points per primitive (for efficiency)
    int max_points_per_primitive = 1000;
    
    /// DST conflict threshold (high conflict may trigger splitting)
    double conflict_threshold = 0.3;
    
    /// Covariance regularization to prevent singularity
    double regularization = 0.001;
    
    /// Lambda for uncertainty weighting in clustering
    double uncertainty_weight_lambda = 1.0;
    
    /// K-means parameters
    int kmeans_max_iter = 100;
    double kmeans_tolerance = 1e-4;
    
    /// Number of semantic classes
    int num_classes = DEFAULT_NUM_CLASSES;
};

// =============================================================================
// Adaptive Kernel Configuration
// =============================================================================

/**
 * @brief Configuration for adaptive anisotropic kernel
 */
struct KernelConfig {
    /// Length scale bounds (meters)
    double length_scale_min = 0.1;
    double length_scale_max = 0.5;
    
    /// Uncertainty gating parameters
    double uncertainty_threshold = 0.7;  ///< Gate threshold (kernel=0 above this)
    double uncertainty_low = 0.3;        ///< Low uncertainty reference
    double gamma = 2.0;                  ///< Gating sharpness
    
    /// Reachability kernel parameter
    double reachability_lambda = 1.0;
    
    /// Confidence weighting parameters
    double confidence_weight_beta = 2.0;
    double entropy_weight_gamma = 1.0;
    
    /// Traversable classes for reachability
    std::set<int> traversable_classes = DEFAULT_TRAVERSABLE_CLASSES;
};

// =============================================================================
// Semantic Map Configuration
// =============================================================================

/**
 * @brief Configuration for semantic map
 */
struct MapConfig {
    /// Frame IDs
    std::string frame_id = "map";
    std::string sensor_frame_id = "camera_color_optical_frame";
    
    /// Map resolution (voxel size in meters)
    double resolution = 0.05;
    
    /// Map origin (minimum corner)
    double origin_x = -10.0;
    double origin_y = -10.0;
    double origin_z = -0.5;
    
    /// Map dimensions (meters)
    double size_x = 20.0;
    double size_y = 20.0;
    double size_z = 3.0;
    
    /// Number of semantic classes
    int num_classes = DEFAULT_NUM_CLASSES;
    
    /// Log-odds bounds
    double log_odds_min = LOG_ODDS_MIN;
    double log_odds_max = LOG_ODDS_MAX;
    
    /// Prior probability (uniform: 1/num_classes)
    double prior_prob = 1.0 / DEFAULT_NUM_CLASSES;
    
    /// Maximum number of cells (memory limit)
    size_t max_cells = 1000000;
    
    /**
     * @brief Get grid dimensions
     */
    void getGridSize(int& nx, int& ny, int& nz) const {
        nx = static_cast<int>(std::ceil(size_x / resolution));
        ny = static_cast<int>(std::ceil(size_y / resolution));
        nz = static_cast<int>(std::ceil(size_z / resolution));
    }
    
    /**
     * @brief Check if position is within map bounds
     */
    bool isInBounds(const Vector3d& pos) const {
        return pos.x() >= origin_x && pos.x() < origin_x + size_x &&
               pos.y() >= origin_y && pos.y() < origin_y + size_y &&
               pos.z() >= origin_z && pos.z() < origin_z + size_z;
    }
    
    /**
     * @brief Convert position to grid indices
     */
    bool positionToIndex(const Vector3d& pos, int& ix, int& iy, int& iz) const {
        if (!isInBounds(pos)) return false;
        ix = static_cast<int>(std::floor((pos.x() - origin_x) / resolution));
        iy = static_cast<int>(std::floor((pos.y() - origin_y) / resolution));
        iz = static_cast<int>(std::floor((pos.z() - origin_z) / resolution));
        return true;
    }
    
    /**
     * @brief Convert grid indices to position (cell center)
     */
    Vector3d indexToPosition(int ix, int iy, int iz) const {
        return Vector3d(
            origin_x + (ix + 0.5) * resolution,
            origin_y + (iy + 0.5) * resolution,
            origin_z + (iz + 0.5) * resolution
        );
    }
};

// =============================================================================
// Navigation Configuration
// =============================================================================

/**
 * @brief Configuration for navigation costmap generation
 */
struct NavigationConfig {
    /// Height range for 2D costmap projection
    double costmap_height_min = 0.0;
    double costmap_height_max = 0.5;
    
    /// Cost values
    int8_t free_cost = 0;
    int8_t unknown_cost = -1;
    int8_t obstacle_cost = 100;
    
    /// Inflation parameters
    double inflation_radius = 0.3;
    double cost_scaling_factor = 10.0;
    
    /// Minimum confidence for cost assignment
    double min_confidence = 0.3;
    
    /// Traversable classes
    std::set<int> traversable_classes = DEFAULT_TRAVERSABLE_CLASSES;
};

// =============================================================================
// Exploration Configuration
// =============================================================================

/**
 * @brief Configuration for EMI-based exploration
 */
struct ExplorationConfig {
    /// Maximum exploration distance
    double max_distance = 10.0;
    
    /// Minimum information gain threshold
    double min_info_gain = 0.1;
    
    /// Maximum number of goals to generate
    int max_goals = 10;
    
    /// Sensor parameters for visibility
    double sensor_range = 6.0;
    double sensor_fov_horizontal = 1.2;  // radians (~69 degrees)
    double sensor_fov_vertical = 0.9;    // radians (~52 degrees)
    
    /// Safety constraints
    double min_obstacle_distance = 0.5;
    
    /// Utility weights
    double weight_info_gain = 1.0;
    double weight_distance = 0.3;
    double weight_uncertainty = 0.5;
    
    /// Frontier detection
    int min_frontier_size = 10;
};

// =============================================================================
// Processing Configuration
// =============================================================================

/**
 * @brief Configuration for processing pipeline
 */
struct ProcessingConfig {
    /// Publishing rates (Hz)
    double map_publish_rate = 2.0;
    double costmap_publish_rate = 5.0;
    double primitive_publish_rate = 10.0;
    
    /// Point cloud processing
    int downsample_factor = 2;
    double voxel_filter_size = 0.02;
    
    /// Threading
    int num_threads = 4;
    bool use_async_processing = true;
    
    /// Sensor parameters
    SensorModel sensor_model;
};

// =============================================================================
// Master Configuration
// =============================================================================

/**
 * @brief Master configuration containing all sub-configurations
 */
struct HESFMConfig {
    UncertaintyConfig uncertainty;
    PrimitiveConfig primitive;
    KernelConfig kernel;
    MapConfig map;
    NavigationConfig navigation;
    ExplorationConfig exploration;
    ProcessingConfig processing;
    
    /**
     * @brief Load configuration from ROS parameter server
     * @param nh NodeHandle with namespace
     */
    void loadFromROS(/* ros::NodeHandle& nh */);
    
    /**
     * @brief Save configuration to YAML file
     * @param filepath Path to output file
     */
    bool saveToYAML(const std::string& filepath) const;
    
    /**
     * @brief Load configuration from YAML file
     * @param filepath Path to input file
     */
    bool loadFromYAML(const std::string& filepath);
};

} // namespace hesfm

#endif // HESFM_CONFIG_H_