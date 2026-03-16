/**
 * @file semantic_map.h
 * @brief Semantic map with log-odds Bayesian update for HESFM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * Implements the semantic map using log-odds representation:
 * h_{t+1} = h_t + Σ_j k̃(x, G_j) · w(U_j) · l^{score}(G_j)
 * 
 * Where:
 * - h: Log-odds vector for each class
 * - k̃: Adaptive kernel value
 * - w: Confidence weight
 * - l^{score}: Log-odds observation scores
 */

#ifndef HESFM_SEMANTIC_MAP_H_
#define HESFM_SEMANTIC_MAP_H_

#include "hesfm/types.h"
#include "hesfm/config.h"
#include "hesfm/adaptive_kernel.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace hesfm {

/**
 * @brief Semantic map with voxel grid storage
 * 
 * Maintains semantic state for each voxel using log-odds representation.
 * Supports efficient updates from Gaussian primitives and queries.
 * 
 * @code
 * MapConfig config;
 * config.resolution = 0.05;
 * 
 * SemanticMap map(config);
 * 
 * // Update with primitives
 * AdaptiveKernel kernel;
 * map.update(primitives, kernel);
 * 
 * // Query
 * auto state = map.query(position);
 * int cls = map.getClass(position);
 * @endcode
 */
class SemanticMap {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /**
     * @brief Default constructor
     */
    SemanticMap();
    
    /**
     * @brief Constructor with configuration
     */
    explicit SemanticMap(const MapConfig& config);
    
    /**
     * @brief Destructor
     */
    ~SemanticMap() = default;
    
    // =========================================================================
    // Map Updates
    // =========================================================================
    
    /**
     * @brief Update map with Gaussian primitives
     * 
     * For each cell in the influence region of each primitive:
     * h_{t+1} = h_t + k̃(x, G) · w(U) · l^{score}(G)
     * 
     * @param primitives Vector of Gaussian primitives
     * @param kernel Adaptive kernel for computing influence
     */
    void update(const std::vector<GaussianPrimitive>& primitives,
                const AdaptiveKernel& kernel);
    
    /**
     * @brief Update map with primitives and custom max_trace
     */
    void update(const std::vector<GaussianPrimitive>& primitives,
                const AdaptiveKernel& kernel,
                double max_trace);
    
    /**
     * @brief Update single cell with observation
     * 
     * @param position Cell position
     * @param class_probs Class probability observation
     * @param weight Observation weight
     */
    void updateCell(const Vector3d& position,
                    const std::vector<double>& class_probs,
                    double weight = 1.0);
    
    /**
     * @brief Apply temporal decay to map
     * 
     * Reduces confidence of cells not recently observed
     * 
     * @param decay_rate Rate of decay (0 = no decay, 1 = full reset)
     * @param current_time Current timestamp
     * @param max_age Maximum age before decay starts
     */
    void applyTemporalDecay(double decay_rate, double current_time, double max_age);
    
    // =========================================================================
    // Map Queries
    // =========================================================================
    
    /**
     * @brief Query semantic state at position
     * @return Optional state if cell exists
     */
    std::optional<SemanticState> query(const Vector3d& position) const;
    
    /**
     * @brief Get predicted class at position
     * @return Class index or -1 if not observed
     */
    int getClass(const Vector3d& position) const;
    
    /**
     * @brief Get class probabilities at position
     */
    VectorXd getProbabilities(const Vector3d& position) const;
    
    /**
     * @brief Get confidence at position
     */
    double getConfidence(const Vector3d& position) const;
    
    /**
     * @brief Get uncertainty at position (1 - confidence)
     */
    double getUncertainty(const Vector3d& position) const;
    
    /**
     * @brief Check if position is traversable
     */
    bool isTraversable(const Vector3d& position) const;
    
    /**
     * @brief Check if position is occupied by obstacle
     */
    bool isObstacle(const Vector3d& position) const;
    
    /**
     * @brief Get map cell at position
     */
    std::optional<MapCell> getCell(const Vector3d& position) const;
    
    /**
     * @brief Get all observed cells
     */
    std::vector<MapCell> getOccupiedCells() const;
    
    /**
     * @brief Get cells within bounding box
     */
    std::vector<MapCell> getCellsInBBox(const Vector3d& min_pt,
                                         const Vector3d& max_pt) const;
    
    /**
     * @brief Get cells of a specific class
     */
    std::vector<MapCell> getCellsByClass(int semantic_class,
                                          double min_confidence = 0.5) const;
    
    // =========================================================================
    // Navigation Interface
    // =========================================================================
    
    /**
     * @brief Generate 2D occupancy grid costmap
     * 
     * Projects 3D semantic map to 2D within height range
     * 
     * @param height_min Minimum height to consider
     * @param height_max Maximum height to consider
     * @param[out] width Costmap width
     * @param[out] height Costmap height
     * @return Costmap data (row-major, values: -1=unknown, 0=free, 100=occupied)
     */
    std::vector<int8_t> generateCostmap(double height_min, double height_max,
                                         int& width, int& height) const;
    
    /**
     * @brief Generate costmap with custom parameters
     */
    std::vector<int8_t> generateCostmap(const NavigationConfig& nav_config,
                                         int& width, int& height) const;
    
    /**
     * @brief Get navigation cost at position
     * @return Cost value (-1=unknown, 0=free, 1-99=inflated, 100=lethal)
     */
    int8_t getNavigationCost(const Vector3d& position) const;
    
    /**
     * @brief Find nearest traversable position
     */
    std::optional<Vector3d> findNearestTraversable(const Vector3d& position,
                                                    double max_distance) const;
    
    // =========================================================================
    // Map Management
    // =========================================================================
    
    /**
     * @brief Reset entire map
     */
    void reset();
    
    /**
     * @brief Reset region of map
     */
    void resetRegion(const Vector3d& min_pt, const Vector3d& max_pt);
    
    /**
     * @brief Prune low-confidence cells
     */
    size_t pruneByConfidence(double min_confidence);
    
    /**
     * @brief Prune cells with few observations
     */
    size_t pruneByObservationCount(int min_observations);
    
    // =========================================================================
    // Map I/O
    // =========================================================================
    
    /**
     * @brief Save map to file
     */
    bool save(const std::string& filepath, const std::string& format = "yaml") const;
    
    /**
     * @brief Load map from file
     */
    bool load(const std::string& filepath);
    
    /**
     * @brief Export as point cloud
     */
    std::vector<std::tuple<Vector3d, int, double>> toPointCloud() const;
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    /**
     * @brief Get number of occupied cells
     */
    size_t getNumCells() const { return cells_.size(); }
    
    /**
     * @brief Get map coverage (fraction of volume observed)
     */
    double getCoverage() const;
    
    /**
     * @brief Get mean confidence across map
     */
    double getMeanConfidence() const;
    
    /**
     * @brief Get mean uncertainty across map
     */
    double getMeanUncertainty() const;
    
    /**
     * @brief Get class distribution
     */
    std::vector<int> getClassDistribution() const;
    
    /**
     * @brief Get memory usage in bytes
     */
    size_t getMemoryUsage() const;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    const MapConfig& getConfig() const { return config_; }
    void setConfig(const MapConfig& config);
    
    /**
     * @brief Set traversable classes
     */
    void setTraversableClasses(const std::set<int>& classes) {
        traversable_classes_ = classes;
    }

private:
    MapConfig config_;
    std::set<int> traversable_classes_;
    
    /// Cell storage: hash -> MapCell
    std::unordered_map<size_t, MapCell> cells_;
    
    /// Mutex for thread safety
    mutable std::shared_mutex mutex_;
    
    /// Statistics
    size_t total_observations_ = 0;
    double last_update_time_ = 0.0;
    
    // =========================================================================
    // Internal Methods
    // =========================================================================
    
    /**
     * @brief Convert position to hash key
     */
    size_t positionToHash(const Vector3d& position) const;
    
    /**
     * @brief Convert position to grid indices
     */
    bool positionToIndex(const Vector3d& position, int& ix, int& iy, int& iz) const;
    
    /**
     * @brief Convert grid indices to position
     */
    Vector3d indexToPosition(int ix, int iy, int iz) const;
    
    /**
     * @brief Get or create cell at position
     */
    MapCell& getOrCreateCell(const Vector3d& position);
    
    /**
     * @brief Compute log-odds from probability
     */
    double probToLogOdds(double prob) const;
    
    /**
     * @brief Compute probability from log-odds
     */
    double logOddsToProb(double log_odds) const;
    
    /**
     * @brief Compute confidence weight for update
     */
    double computeConfidenceWeight(double uncertainty, double entropy) const;
    
    /**
     * @brief Clamp log-odds to valid range
     */
    double clampLogOdds(double log_odds) const;
};

} // namespace hesfm

#endif // HESFM_SEMANTIC_MAP_H_
