/**
 * @file adaptive_kernel.cpp
 * @brief Implementation of adaptive anisotropic kernel
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include "hesfm/adaptive_kernel.h"
#include <cmath>
#include <algorithm>

namespace hesfm {

// =============================================================================
// Constructors
// =============================================================================

AdaptiveKernel::AdaptiveKernel()
    : config_() {}

AdaptiveKernel::AdaptiveKernel(const KernelConfig& config)
    : config_(config) {}

// =============================================================================
// Main Kernel Interface
// =============================================================================

double AdaptiveKernel::compute(const Vector3d& query_point,
                                const GaussianPrimitive& primitive,
                                double max_trace) const {
    
    // Binary filter: reject high uncertainty primitives
    if (filterFunction(primitive.uncertainty) < 0.5) {
        return 0.0;
    }
    
    // Compute adaptive length scale
    double length_scale = computeAdaptiveLengthScale(primitive.covariance, max_trace);
    
    // Geometric kernel
    double k_geo = geometricKernel(query_point, primitive, length_scale);
    if (k_geo < EPSILON) {
        return 0.0;
    }
    
    // Uncertainty gating
    double k_unc = uncertaintyKernel(primitive.uncertainty);
    
    // Reachability kernel
    double distance = (query_point - primitive.centroid).norm();
    double k_reach = reachabilityKernel(primitive.semantic_class, distance);
    
    // Combined kernel
    return k_geo * k_unc * k_reach;
}

std::vector<double> AdaptiveKernel::computeBatch(
    const std::vector<Vector3d>& query_points,
    const GaussianPrimitive& primitive,
    double max_trace) const {
    
    std::vector<double> kernel_values(query_points.size());
    
    // Pre-compute common values
    double length_scale = computeAdaptiveLengthScale(primitive.covariance, max_trace);
    double k_unc = uncertaintyKernel(primitive.uncertainty);
    double filter = filterFunction(primitive.uncertainty);
    
    if (filter < 0.5) {
        std::fill(kernel_values.begin(), kernel_values.end(), 0.0);
        return kernel_values;
    }
    
    // Pre-compute inverse covariance
    Matrix3d cov_inv = safeInverse(primitive.covariance);
    
    #pragma omp parallel for
    for (size_t i = 0; i < query_points.size(); ++i) {
        // Mahalanobis distance
        Vector3d diff = query_points[i] - primitive.centroid;
        double d_M = std::sqrt(diff.transpose() * cov_inv * diff);
        
        // Sparse kernel
        double k_geo = sparseKernel(d_M, length_scale);
        
        if (k_geo < EPSILON) {
            kernel_values[i] = 0.0;
            continue;
        }
        
        // Reachability
        double distance = diff.norm();
        double k_reach = reachabilityKernel(primitive.semantic_class, distance);
        
        kernel_values[i] = k_geo * k_unc * k_reach;
    }
    
    return kernel_values;
}

MatrixXd AdaptiveKernel::computeKernelMatrix(
    const std::vector<GaussianPrimitive>& primitives,
    const std::vector<Vector3d>& cell_positions,
    double max_trace) const {
    
    MatrixXd K(primitives.size(), cell_positions.size());
    K.setZero();
    
    #pragma omp parallel for
    for (size_t j = 0; j < primitives.size(); ++j) {
        auto row = computeBatch(cell_positions, primitives[j], max_trace);
        for (size_t i = 0; i < cell_positions.size(); ++i) {
            K(j, i) = row[i];
        }
    }
    
    return K;
}

// =============================================================================
// Individual Kernel Components
// =============================================================================

double AdaptiveKernel::sparseKernel(double distance, double length_scale) const {
    // Compact support: kernel is zero outside length scale
    if (distance >= length_scale) {
        return 0.0;
    }
    
    double d_norm = distance / length_scale;
    double two_pi = 2.0 * M_PI;
    
    // Wendland's compactly supported kernel (C2 smooth)
    // k'(d, l) = ((2 + cos(2πd/l))(1 - d/l))/3 + sin(2πd/l)/(2π)
    double term1 = (2.0 + std::cos(two_pi * d_norm)) * (1.0 - d_norm) / 3.0;
    double term2 = std::sin(two_pi * d_norm) / two_pi;
    
    return std::max(0.0, term1 + term2);
}

double AdaptiveKernel::rbfKernel(double distance, double length_scale) const {
    // Gaussian RBF kernel
    // k_rbf(d, l) = exp(-d²/(2l²))
    double scaled_dist = distance / length_scale;
    return std::exp(-0.5 * scaled_dist * scaled_dist);
}

double AdaptiveKernel::geometricKernel(const Vector3d& query,
                                        const GaussianPrimitive& primitive,
                                        double length_scale) const {
    // Mahalanobis distance using primitive covariance
    double d_M = mahalanobisDistance(query, primitive.centroid, primitive.covariance);
    
    // Apply sparse kernel
    return sparseKernel(d_M, length_scale);
}

double AdaptiveKernel::uncertaintyKernel(double uncertainty) const {
    // Binary threshold check
    if (uncertainty >= config_.uncertainty_threshold) {
        return 0.0;
    }
    
    // Gaussian weighting centered at low uncertainty
    double diff = uncertainty - config_.uncertainty_low;
    return std::exp(-config_.gamma * diff * diff);
}

double AdaptiveKernel::reachabilityKernel(int semantic_class, double distance) const {
    // Traversable classes get full influence
    if (config_.traversable_classes.count(semantic_class) > 0) {
        return 1.0;
    }
    
    // Non-traversable: exponential decay with distance
    return std::exp(-config_.reachability_lambda * distance);
}

double AdaptiveKernel::filterFunction(double uncertainty) const {
    // Binary filter
    return (uncertainty < config_.uncertainty_threshold) ? 1.0 : 0.0;
}

// =============================================================================
// Length Scale Computation
// =============================================================================

double AdaptiveKernel::computeAdaptiveLengthScale(
    const Matrix3d& covariance,
    double max_trace) const {
    
    if (max_trace < EPSILON) {
        return config_.length_scale_min;
    }
    
    double trace = covariance.trace();
    
    // Cube root scaling for 3D
    double ratio = std::pow(trace / max_trace, 1.0 / 3.0);
    ratio = std::clamp(ratio, 0.0, 1.0);
    
    return config_.length_scale_min + 
           (config_.length_scale_max - config_.length_scale_min) * ratio;
}

std::vector<double> AdaptiveKernel::computeAllLengthScales(
    const std::vector<GaussianPrimitive>& primitives,
    double max_trace) const {
    
    std::vector<double> scales(primitives.size());
    
    for (size_t i = 0; i < primitives.size(); ++i) {
        scales[i] = computeAdaptiveLengthScale(primitives[i].covariance, max_trace);
    }
    
    return scales;
}

// =============================================================================
// Confidence Weighting
// =============================================================================

double AdaptiveKernel::computeConfidenceWeight(
    double uncertainty,
    double entropy,
    double max_entropy) const {
    
    // Weight decreases with uncertainty
    double w_unc = std::pow(1.0 - uncertainty, config_.confidence_weight_beta);
    
    // Weight decreases with entropy
    double normalized_entropy = (max_entropy > EPSILON) ? (entropy / max_entropy) : 0.0;
    double w_ent = std::pow(1.0 - normalized_entropy, config_.entropy_weight_gamma);
    
    return w_unc * w_ent;
}

double AdaptiveKernel::computeConfidenceWeight(double uncertainty) const {
    // Simplified version using only uncertainty
    return std::pow(1.0 - uncertainty, config_.confidence_weight_beta);
}

// =============================================================================
// Utility Methods
// =============================================================================

double AdaptiveKernel::mahalanobisDistance(
    const Vector3d& point,
    const Vector3d& mean,
    const Matrix3d& covariance) const {
    
    Vector3d diff = point - mean;
    Matrix3d cov_inv = safeInverse(covariance);
    
    double d_sq = diff.transpose() * cov_inv * diff;
    return std::sqrt(std::max(0.0, d_sq));
}

bool AdaptiveKernel::isInSupport(const Vector3d& query,
                                  const GaussianPrimitive& primitive,
                                  double length_scale) const {
    
    double d_M = mahalanobisDistance(query, primitive.centroid, primitive.covariance);
    return d_M < length_scale;
}

void AdaptiveKernel::getInfluenceBounds(
    const GaussianPrimitive& primitive,
    double length_scale,
    Vector3d& min_bound,
    Vector3d& max_bound) const {
    
    // Get eigenvalues for scaling
    Eigen::SelfAdjointEigenSolver<Matrix3d> solver(primitive.covariance);
    Vector3d eigenvalues = solver.eigenvalues();
    
    // Maximum extent in each direction
    Vector3d extent;
    for (int i = 0; i < 3; ++i) {
        extent(i) = length_scale * std::sqrt(std::max(eigenvalues(i), EPSILON));
    }
    
    // Note: This is conservative (axis-aligned bound of rotated ellipsoid)
    // For tighter bounds, would need to consider eigenvector rotation
    min_bound = primitive.centroid - extent;
    max_bound = primitive.centroid + extent;
}

// =============================================================================
// Private Methods
// =============================================================================

Matrix3d AdaptiveKernel::safeInverse(const Matrix3d& covariance) const {
    // Add regularization for numerical stability
    Matrix3d regularized = covariance + EPSILON * Matrix3d::Identity();
    
    // Check condition number
    Eigen::JacobiSVD<Matrix3d> svd(regularized);
    double cond = svd.singularValues()(0) / svd.singularValues()(2);
    
    if (cond > 1e6) {
        // Ill-conditioned: add more regularization
        regularized = covariance + 0.001 * Matrix3d::Identity();
    }
    
    return regularized.inverse();
}

} // namespace hesfm
