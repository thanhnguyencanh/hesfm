/**
 * @file adaptive_kernel.h
 * @brief Adaptive anisotropic kernel for HESFM BKI
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * Implements the composite kernel function:
 * k̃(x, G_j) = k_geo(x, μ_j, Σ_j, l_j) · k_unc(u_j) · k_reach(c_j, d) · f_filter(u_j)
 * 
 * Components:
 * - k_geo: Geometric kernel using Mahalanobis distance with sparse support
 * - k_unc: Uncertainty gating function
 * - k_reach: Reachability-based influence
 * - f_filter: Binary filter for high uncertainty
 */

#ifndef HESFM_ADAPTIVE_KERNEL_H_
#define HESFM_ADAPTIVE_KERNEL_H_

#include "hesfm/types.h"
#include "hesfm/config.h"

namespace hesfm {

/**
 * @brief Adaptive anisotropic kernel module
 * 
 * Computes kernel values between query positions and Gaussian primitives
 * for Bayesian Kernel Inference map updates.
 * 
 * @code
 * KernelConfig config;
 * config.length_scale_min = 0.1;
 * config.length_scale_max = 0.5;
 * 
 * AdaptiveKernel kernel(config);
 * 
 * Vector3d query = ...;
 * GaussianPrimitive primitive = ...;
 * double k_val = kernel.compute(query, primitive, max_trace);
 * @endcode
 */
class AdaptiveKernel {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /**
     * @brief Default constructor
     */
    AdaptiveKernel();
    
    /**
     * @brief Constructor with configuration
     */
    explicit AdaptiveKernel(const KernelConfig& config);
    
    /**
     * @brief Destructor
     */
    ~AdaptiveKernel() = default;
    
    // =========================================================================
    // Main Kernel Interface
    // =========================================================================
    
    /**
     * @brief Compute full adaptive kernel value
     * 
     * k̃(x, G) = k_geo · k_unc · k_reach · f_filter
     * 
     * @param query_point Query position
     * @param primitive Gaussian primitive
     * @param max_trace Maximum covariance trace for length scale normalization
     * @return Kernel value in [0, 1]
     */
    double compute(const Vector3d& query_point,
                   const GaussianPrimitive& primitive,
                   double max_trace) const;
    
    /**
     * @brief Compute kernel for multiple query points
     * 
     * @param query_points Vector of query positions
     * @param primitive Gaussian primitive
     * @param max_trace Maximum covariance trace
     * @return Vector of kernel values
     */
    std::vector<double> computeBatch(const std::vector<Vector3d>& query_points,
                                      const GaussianPrimitive& primitive,
                                      double max_trace) const;
    
    /**
     * @brief Compute kernel for all primitive-cell pairs
     * 
     * @param primitives Vector of Gaussian primitives
     * @param cell_positions Vector of cell center positions
     * @param max_trace Maximum covariance trace
     * @return Matrix of kernel values [num_primitives x num_cells]
     */
    MatrixXd computeKernelMatrix(const std::vector<GaussianPrimitive>& primitives,
                                  const std::vector<Vector3d>& cell_positions,
                                  double max_trace) const;
    
    // =========================================================================
    // Individual Kernel Components
    // =========================================================================
    
    /**
     * @brief Sparse kernel function with compact support
     * 
     * Wendland's compactly supported kernel:
     * k'(d, l) = ((2 + cos(2πd/l))(1 - d/l))/3 + sin(2πd/l)/(2π)  for d < l
     * k'(d, l) = 0  for d >= l
     * 
     * @param distance Distance value
     * @param length_scale Length scale parameter
     * @return Kernel value in [0, 1]
     */
    double sparseKernel(double distance, double length_scale) const;
    
    /**
     * @brief RBF (Gaussian) kernel
     * 
     * k_rbf(d, l) = exp(-d²/(2l²))
     * 
     * @param distance Distance value
     * @param length_scale Length scale parameter
     * @return Kernel value in [0, 1]
     */
    double rbfKernel(double distance, double length_scale) const;
    
    /**
     * @brief Geometric kernel using Mahalanobis distance
     * 
     * d_M = √((x - μ)ᵀ Σ⁻¹ (x - μ))
     * k_geo = sparse_kernel(d_M, l)
     * 
     * @param query Query position
     * @param primitive Gaussian primitive (centroid + covariance)
     * @param length_scale Adaptive length scale
     * @return Kernel value in [0, 1]
     */
    double geometricKernel(const Vector3d& query,
                           const GaussianPrimitive& primitive,
                           double length_scale) const;
    
    /**
     * @brief Uncertainty gating kernel
     * 
     * k_unc(u) = exp(-γ(u - u_low)²)  for u < u_threshold
     * k_unc(u) = 0                     for u >= u_threshold
     * 
     * @param uncertainty Uncertainty value
     * @return Gating weight in [0, 1]
     */
    double uncertaintyKernel(double uncertainty) const;
    
    /**
     * @brief Reachability-based kernel
     * 
     * k_reach(c, d) = 1                        if c ∈ traversable
     * k_reach(c, d) = exp(-λ_reach * d)        otherwise
     * 
     * @param semantic_class Semantic class of primitive
     * @param distance Euclidean distance to query
     * @return Reachability weight in [0, 1]
     */
    double reachabilityKernel(int semantic_class, double distance) const;
    
    /**
     * @brief Binary filter for high uncertainty
     * 
     * f_filter(u) = 1 if u < u_threshold, 0 otherwise
     * 
     * @param uncertainty Uncertainty value
     * @return 1 or 0
     */
    double filterFunction(double uncertainty) const;
    
    // =========================================================================
    // Length Scale Computation
    // =========================================================================
    
    /**
     * @brief Compute adaptive length scale for a primitive
     * 
     * l = l_min + (l_max - l_min) * (trace(Σ) / max_trace)^(1/3)
     * 
     * Larger primitives (higher trace) get larger length scales
     * for smoother influence regions.
     * 
     * @param covariance Covariance matrix
     * @param max_trace Maximum trace for normalization
     * @return Adaptive length scale
     */
    double computeAdaptiveLengthScale(const Matrix3d& covariance, double max_trace) const;

    /**
     * @brief Compute uncertainty-adaptive length scale (EvSemMap-inspired)
     *
     * Extends base geometric length scale with uncertainty-driven expansion:
     * l_adaptive = l_base * exp(1 + b * uncertainty)
     *
     * High-uncertainty regions get larger kernels so nearby certain
     * observations can propagate further, while low-uncertainty regions
     * keep tight, precise kernels.
     *
     * @param covariance Covariance matrix
     * @param max_trace Maximum trace for normalization
     * @param uncertainty Primitive uncertainty [0, 1]
     * @return Adaptive length scale
     */
    double computeUncertaintyAdaptiveLengthScale(
        const Matrix3d& covariance, double max_trace, double uncertainty) const;

    /**
     * @brief Compute length scales for all primitives
     */
    std::vector<double> computeAllLengthScales(
        const std::vector<GaussianPrimitive>& primitives,
        double max_trace) const;
    
    // =========================================================================
    // Confidence Weighting
    // =========================================================================
    
    /**
     * @brief Compute confidence weight for map update
     * 
     * w(u, H) = (1 - u)^β * (1 - H/H_max)^γ
     * 
     * @param uncertainty Primitive uncertainty
     * @param entropy Entropy of class distribution
     * @param max_entropy Maximum possible entropy
     * @return Confidence weight
     */
    double computeConfidenceWeight(double uncertainty,
                                    double entropy,
                                    double max_entropy) const;
    
    /**
     * @brief Simplified confidence weight using only uncertainty
     */
    double computeConfidenceWeight(double uncertainty) const;
    
    // =========================================================================
    // Utility Methods
    // =========================================================================
    
    /**
     * @brief Compute Mahalanobis distance
     */
    double mahalanobisDistance(const Vector3d& point,
                                const Vector3d& mean,
                                const Matrix3d& covariance) const;
    
    /**
     * @brief Check if query is within kernel support
     */
    bool isInSupport(const Vector3d& query,
                     const GaussianPrimitive& primitive,
                     double length_scale) const;
    
    /**
     * @brief Get bounding box of kernel influence
     */
    void getInfluenceBounds(const GaussianPrimitive& primitive,
                            double length_scale,
                            Vector3d& min_bound,
                            Vector3d& max_bound) const;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    const KernelConfig& getConfig() const { return config_; }
    void setConfig(const KernelConfig& config) { config_ = config; }
    
    void setLengthScaleBounds(double l_min, double l_max) {
        config_.length_scale_min = l_min;
        config_.length_scale_max = l_max;
    }
    
    void setUncertaintyThreshold(double threshold) {
        config_.uncertainty_threshold = threshold;
    }
    
    void setTraversableClasses(const std::set<int>& classes) {
        config_.traversable_classes = classes;
    }

private:
    KernelConfig config_;
    
    /**
     * @brief Safely invert covariance matrix with regularization
     */
    Matrix3d safeInverse(const Matrix3d& covariance) const;
};

} // namespace hesfm

#endif // HESFM_ADAPTIVE_KERNEL_H_
