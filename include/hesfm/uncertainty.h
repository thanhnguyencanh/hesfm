/**
 * @file uncertainty.h
 * @brief Multi-source uncertainty decomposition for HESFM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * Implements the uncertainty decomposition:
 * U_total = w_sem * U_sem + w_spa * U_spa + w_obs * U_obs + w_temp * U_temp
 * 
 * Where:
 * - U_sem: Semantic uncertainty from Evidential Deep Learning
 * - U_spa: Spatial consistency uncertainty
 * - U_obs: Observation/sensor model uncertainty
 * - U_temp: Temporal prediction consistency uncertainty
 */

#ifndef HESFM_UNCERTAINTY_H_
#define HESFM_UNCERTAINTY_H_

#include "hesfm/types.h"
#include "hesfm/config.h"
#include <unordered_map>
#include <memory>
#include <mutex>

namespace hesfm {

/**
 * @brief Multi-source uncertainty decomposition module
 * 
 * This class computes individual uncertainty components and combines them
 * into a total uncertainty value for each semantic point.
 * 
 * @code
 * UncertaintyConfig config;
 * config.w_semantic = 0.4;
 * config.w_spatial = 0.2;
 * 
 * UncertaintyDecomposer decomposer(config);
 * 
 * // Process points
 * std::vector<SemanticPoint> points = ...;
 * Vector3d sensor_origin = ...;
 * decomposer.processPointCloud(points, sensor_origin);
 * @endcode
 */
class UncertaintyDecomposer {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /**
     * @brief Default constructor with default configuration
     */
    UncertaintyDecomposer();
    
    /**
     * @brief Constructor with configuration
     * @param config Uncertainty configuration
     */
    explicit UncertaintyDecomposer(const UncertaintyConfig& config);
    
    /**
     * @brief Destructor
     */
    ~UncertaintyDecomposer() = default;
    
    // =========================================================================
    // Individual Uncertainty Components
    // =========================================================================
    
    /**
     * @brief Compute semantic uncertainty from EDL evidence
     * 
     * Using Evidential Deep Learning formulation:
     * α = evidence + 1 (Dirichlet parameters)
     * S = Σα (Dirichlet strength)
     * U_sem = K / S (where K = number of classes)
     * 
     * @param evidence Evidence vector from EDL network
     * @param num_classes Number of semantic classes
     * @return Semantic uncertainty in [0, 1]
     */
    double computeSemanticUncertainty(const std::vector<double>& evidence,
                                       int num_classes) const;
    
    /**
     * @brief Compute semantic uncertainty from class probabilities
     * 
     * Alternative using entropy-based measure when evidence not available:
     * U_sem = H(p) / H_max = -Σp*log(p) / log(K)
     * 
     * @param probabilities Class probability distribution
     * @return Semantic uncertainty in [0, 1]
     */
    double computeSemanticUncertaintyFromProbs(const std::vector<double>& probabilities) const;
    
    /**
     * @brief Compute spatial uncertainty from local consistency
     * 
     * U_spa = 1 - (same_class_neighbors / total_neighbors)
     * High uncertainty when neighbors disagree on class
     * 
     * @param point Query point
     * @param neighbors Neighboring points
     * @return Spatial uncertainty in [0, 1]
     */
    double computeSpatialUncertainty(const SemanticPoint& point,
                                      const std::vector<SemanticPoint>& neighbors) const;
    
    /**
     * @brief Compute observation uncertainty from sensor model
     * 
     * U_obs = σ_r * (r/r_max) + σ_ρ * (1 - ρ/ρ_max) + σ_θ * |cos(θ)|
     * 
     * Components:
     * - Range: Uncertainty increases with distance
     * - Density: Uncertainty increases in sparse regions
     * - Incidence: Uncertainty increases at grazing angles
     * 
     * @param point Point position
     * @param sensor_origin Sensor position
     * @param local_density Local point density (-1 to skip)
     * @param surface_normal Surface normal (zero vector to skip)
     * @return Observation uncertainty in [0, 1]
     */
    double computeObservationUncertainty(const Vector3d& point,
                                          const Vector3d& sensor_origin,
                                          double local_density = -1.0,
                                          const Vector3d& surface_normal = Vector3d::Zero()) const;
    
    /**
     * @brief Compute temporal uncertainty from prediction history
     * 
     * U_temp = 1 - max_c(count_c) / total_observations
     * Low uncertainty when predictions are consistent over time
     * 
     * @param position Spatial position (used as key for history)
     * @param current_class Current predicted class
     * @return Temporal uncertainty in [0, 1]
     */
    double computeTemporalUncertainty(const Vector3d& position, int current_class);
    
    // =========================================================================
    // Combined Uncertainty
    // =========================================================================
    
    /**
     * @brief Compute total uncertainty as weighted sum
     * 
     * U_total = w_sem*U_sem + w_spa*U_spa + w_obs*U_obs + w_temp*U_temp
     * 
     * @param u_sem Semantic uncertainty
     * @param u_spa Spatial uncertainty
     * @param u_obs Observation uncertainty
     * @param u_temp Temporal uncertainty
     * @return Total uncertainty in [0, 1]
     */
    double computeTotalUncertainty(double u_sem, double u_spa,
                                    double u_obs, double u_temp) const;
    
    /**
     * @brief Full uncertainty decomposition for a single point
     * 
     * @param point Query point with semantic info
     * @param neighbors Neighboring points for spatial consistency
     * @param sensor_origin Sensor position
     * @return Complete uncertainty decomposition
     */
    UncertaintyDecomposition decompose(const SemanticPoint& point,
                                        const std::vector<SemanticPoint>& neighbors,
                                        const Vector3d& sensor_origin);
    
    // =========================================================================
    // Batch Processing
    // =========================================================================
    
    /**
     * @brief Process entire point cloud and compute uncertainties
     * 
     * Builds KD-tree for efficient neighbor queries and computes
     * all uncertainty components for each point.
     * 
     * @param[in,out] points Points to process (uncertainties updated in-place)
     * @param sensor_origin Sensor position
     */
    void processPointCloud(std::vector<SemanticPoint>& points,
                           const Vector3d& sensor_origin);
    
    /**
     * @brief Process point cloud with external neighbor structure
     * 
     * @param[in,out] points Points to process
     * @param neighbor_indices Pre-computed neighbor indices for each point
     * @param sensor_origin Sensor position
     */
    void processPointCloudWithNeighbors(
        std::vector<SemanticPoint>& points,
        const std::vector<std::vector<int>>& neighbor_indices,
        const Vector3d& sensor_origin);
    
    // =========================================================================
    // History Management
    // =========================================================================
    
    /**
     * @brief Reset temporal history
     */
    void resetTemporalHistory();
    
    /**
     * @brief Clear temporal history for a region
     * @param min_pt Minimum corner of region
     * @param max_pt Maximum corner of region
     */
    void clearTemporalHistoryInRegion(const Vector3d& min_pt, const Vector3d& max_pt);
    
    /**
     * @brief Get number of tracked positions
     */
    size_t getTemporalHistorySize() const;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    /**
     * @brief Get current configuration
     */
    const UncertaintyConfig& getConfig() const { return config_; }
    
    /**
     * @brief Update configuration
     */
    void setConfig(const UncertaintyConfig& config) { config_ = config; }
    
    /**
     * @brief Update individual weights
     */
    void setWeights(double w_sem, double w_spa, double w_obs, double w_temp);

private:
    /// Configuration
    UncertaintyConfig config_;
    
    /// Sensor model
    SensorModel sensor_model_;
    
    /// Temporal history: spatial hash -> class observation counts
    struct TemporalHistory {
        std::vector<int> class_counts;
        int total_observations = 0;
        
        TemporalHistory(int num_classes = DEFAULT_NUM_CLASSES)
            : class_counts(num_classes, 0) {}
    };
    std::unordered_map<size_t, TemporalHistory> temporal_history_;
    
    /// Mutex for thread safety
    mutable std::mutex mutex_;
    
    /**
     * @brief Compute spatial hash for position
     */
    size_t spatialHash(const Vector3d& position) const;
    
    /**
     * @brief Find neighbors within radius
     */
    std::vector<int> findNeighbors(const std::vector<SemanticPoint>& points,
                                    int query_idx, double radius) const;
    
    /**
     * @brief Compute local point density
     */
    double computeLocalDensity(const std::vector<SemanticPoint>& points,
                                int query_idx, double radius) const;
};

} // namespace hesfm

#endif // HESFM_UNCERTAINTY_H_