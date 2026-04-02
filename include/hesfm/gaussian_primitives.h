/**
 * @file gaussian_primitives.h
 * @brief Hierarchical Gaussian primitive construction for HESFM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * Implements the hierarchical aggregation of semantic points into
 * Gaussian primitives using:
 * - Uncertainty-weighted K-means++ clustering
 * - Dempster-Shafer Theory (DST) fusion for semantic aggregation
 * - Conflict-aware primitive refinement
 */

#ifndef HESFM_GAUSSIAN_PRIMITIVES_H_
#define HESFM_GAUSSIAN_PRIMITIVES_H_

#include "hesfm/types.h"
#include "hesfm/config.h"
#include <memory>
#include <random>

namespace hesfm {

/**
 * @brief Gaussian primitive construction module
 * 
 * Aggregates point-level semantics into compact Gaussian primitives
 * for efficient map updates.
 * 
 * @code
 * PrimitiveConfig config;
 * config.target_primitives = 128;
 * 
 * GaussianPrimitiveBuilder builder(config);
 * 
 * std::vector<SemanticPoint> points = ...;
 * auto primitives = builder.buildPrimitives(points);
 * @endcode
 */
class GaussianPrimitiveBuilder {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /**
     * @brief Default constructor
     */
    GaussianPrimitiveBuilder();
    
    /**
     * @brief Constructor with configuration
     */
    explicit GaussianPrimitiveBuilder(const PrimitiveConfig& config);
    
    /**
     * @brief Destructor
     */
    ~GaussianPrimitiveBuilder() = default;
    
    // =========================================================================
    // Main Building Interface
    // =========================================================================
    
    /**
     * @brief Build Gaussian primitives from semantic points
     * 
     * Pipeline:
     * 1. Uncertainty-weighted K-means++ clustering
     * 2. Compute centroid and covariance for each cluster
     * 3. DST fusion for semantic aggregation
     * 4. Conflict checking and filtering
     * 
     * @param points Input semantic points with uncertainties
     * @return Vector of Gaussian primitives
     */
    std::vector<GaussianPrimitive> buildPrimitives(const std::vector<SemanticPoint>& points);
    
    /**
     * @brief Build primitives with automatic cluster count
     * 
     * Determines optimal number of clusters based on point distribution
     * 
     * @param points Input points
     * @param min_primitives Minimum number of primitives
     * @param max_primitives Maximum number of primitives
     * @return Vector of primitives
     */
    std::vector<GaussianPrimitive> buildPrimitivesAuto(
        const std::vector<SemanticPoint>& points,
        int min_primitives = 16,
        int max_primitives = 256);
    
    // =========================================================================
    // Incremental Updates
    // =========================================================================
    
    /**
     * @brief Update existing primitives with new observations
     * 
     * Associates new points with existing primitives and updates
     * statistics incrementally.
     * 
     * @param primitives Existing primitives to update
     * @param new_points New semantic points
     * @param max_distance Maximum association distance
     * @return Updated primitives
     */
    std::vector<GaussianPrimitive> updatePrimitives(
        const std::vector<GaussianPrimitive>& primitives,
        const std::vector<SemanticPoint>& new_points,
        double max_distance = 0.5);
    
    /**
     * @brief Merge two primitives
     * 
     * @param prim1 First primitive
     * @param prim2 Second primitive
     * @return Merged primitive
     */
    GaussianPrimitive mergePrimitives(const GaussianPrimitive& prim1,
                                       const GaussianPrimitive& prim2);
    
    /**
     * @brief Split a primitive with high conflict
     * 
     * @param primitive Primitive to split
     * @param points Points belonging to this primitive
     * @param num_splits Number of sub-primitives
     * @return Split primitives
     */
    std::vector<GaussianPrimitive> splitPrimitive(
        const GaussianPrimitive& primitive,
        const std::vector<SemanticPoint>& points,
        int num_splits = 2);
    
    // =========================================================================
    // Dempster-Shafer Fusion
    // =========================================================================
    
    /**
     * @brief Fuse two belief distributions using Dempster's rule
     * 
     * Combined belief: m_12(A) = [Σ_{B∩C=A} m1(B)*m2(C)] / (1 - K)
     * Conflict: K = Σ_{B∩C=∅} m1(B)*m2(C)
     * 
     * @param belief1 First belief mass function
     * @param belief2 Second belief mass function
     * @param uncertainty1 First source uncertainty (used as mass on Θ)
     * @param uncertainty2 Second source uncertainty
     * @param[out] conflict Computed conflict value
     * @return Fused belief distribution
     */
    std::vector<double> dstFusion(const std::vector<double>& belief1,
                                   const std::vector<double>& belief2,
                                   double uncertainty1,
                                   double uncertainty2,
                                   double& conflict);
    
    /**
     * @brief Fuse multiple belief distributions
     * 
     * @param beliefs Vector of belief distributions
     * @param uncertainties Vector of uncertainties
     * @param[out] total_conflict Maximum pairwise conflict
     * @return Fused belief distribution
     */
    std::vector<double> dstFusionMultiple(
        const std::vector<std::vector<double>>& beliefs,
        const std::vector<double>& uncertainties,
        double& total_conflict);
    
    // =========================================================================
    // Clustering Methods
    // =========================================================================
    
    /**
     * @brief Uncertainty-weighted K-means++ initialization
     * 
     * Selects initial centroids with probability proportional to
     * squared distance weighted by inverse uncertainty.
     * 
     * @param points Input points
     * @param k Number of clusters
     * @return Initial centroid positions
     */
    std::vector<Vector3d> kmeansppInit(const std::vector<SemanticPoint>& points, int k);
    
    /**
     * @brief Run uncertainty-weighted K-means clustering
     * 
     * @param points Input points
     * @param k Number of clusters
     * @param[out] centroids Final centroid positions
     * @return Cluster assignments for each point
     */
    std::vector<int> uncertaintyWeightedKMeans(
        const std::vector<SemanticPoint>& points,
        int k,
        std::vector<Vector3d>& centroids);
    
    // =========================================================================
    // Adaptive Merge/Split
    // =========================================================================

    /**
     * @brief Refine a set of primitives by merging nearby same-class
     *        primitives and splitting high-conflict ones.
     *
     * @param primitives     Input primitives
     * @param points         Original points (needed for splitting)
     * @param merge_distance Euclidean threshold for merge candidates
     * @return Refined primitives
     */
    std::vector<GaussianPrimitive> refinePrimitives(
        const std::vector<GaussianPrimitive>& primitives,
        const std::vector<SemanticPoint>& points,
        double merge_distance = 0.3);

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /**
     * @brief Compute maximum covariance trace (for length scale normalization)
     */
    double computeMaxTrace(const std::vector<GaussianPrimitive>& primitives) const;
    
    /**
     * @brief Filter primitives by conflict threshold
     */
    std::vector<GaussianPrimitive> filterByConflict(
        const std::vector<GaussianPrimitive>& primitives,
        double threshold) const;
    
    /**
     * @brief Compute statistics about primitives
     */
    void computeStatistics(const std::vector<GaussianPrimitive>& primitives,
                           double& mean_points,
                           double& mean_uncertainty,
                           double& mean_conflict) const;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    const PrimitiveConfig& getConfig() const { return config_; }
    void setConfig(const PrimitiveConfig& config) { config_ = config; }

private:
    PrimitiveConfig config_;
    std::mt19937 rng_;
    uint32_t next_primitive_id_ = 0;
    
    /**
     * @brief Compute primitive from cluster of points
     */
    GaussianPrimitive computePrimitive(const std::vector<SemanticPoint>& points,
                                        const std::vector<int>& indices);
    
    /**
     * @brief Compute weighted centroid
     */
    Vector3d computeWeightedCentroid(const std::vector<SemanticPoint>& points,
                                      const std::vector<int>& indices,
                                      const std::vector<double>& weights);
    
    /**
     * @brief Compute weighted covariance
     */
    Matrix3d computeWeightedCovariance(const std::vector<SemanticPoint>& points,
                                        const std::vector<int>& indices,
                                        const std::vector<double>& weights,
                                        const Vector3d& centroid);
    
    /**
     * @brief Compute point weights (inverse uncertainty)
     */
    std::vector<double> computePointWeights(const std::vector<SemanticPoint>& points,
                                             const std::vector<int>& indices);
    
    /**
     * @brief Get next primitive ID
     */
    uint32_t getNextPrimitiveId() { return next_primitive_id_++; }
};

} // namespace hesfm

#endif // HESFM_GAUSSIAN_PRIMITIVES_H_
