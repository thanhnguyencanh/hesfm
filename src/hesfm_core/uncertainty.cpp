/**
 * @file uncertainty.cpp
 * @brief Implementation of multi-source uncertainty decomposition
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include "hesfm/uncertainty.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace hesfm {

// =============================================================================
// Constructors
// =============================================================================

UncertaintyDecomposer::UncertaintyDecomposer()
    : config_() {}

UncertaintyDecomposer::UncertaintyDecomposer(const UncertaintyConfig& config)
    : config_(config) {}

// =============================================================================
// Semantic Uncertainty
// =============================================================================

double UncertaintyDecomposer::computeSemanticUncertainty(
    const std::vector<double>& evidence,
    int num_classes) const {
    
    if (evidence.empty() || num_classes <= 0) {
        return 1.0;
    }
    
    // Dirichlet parameters: alpha_k = evidence_k + 1
    // Dirichlet strength: S = sum(alpha)
    double S = 0.0;
    for (const double& e : evidence) {
        S += (e + 1.0);
    }
    
    // Uncertainty: U = K / S
    // When S is large (strong evidence), uncertainty is low
    // When S approaches K (uniform prior), uncertainty is high
    double uncertainty = static_cast<double>(num_classes) / S;
    
    return std::clamp(uncertainty, 0.0, 1.0);
}

double UncertaintyDecomposer::computeSemanticUncertaintyFromProbs(
    const std::vector<double>& probabilities) const {
    
    if (probabilities.empty()) {
        return 1.0;
    }
    
    // Compute entropy: H = -sum(p * log(p))
    double entropy = 0.0;
    for (const double& p : probabilities) {
        if (p > EPSILON) {
            entropy -= p * std::log(p);
        }
    }
    
    // Normalize by maximum entropy: H_max = log(K)
    double max_entropy = std::log(static_cast<double>(probabilities.size()));
    
    if (max_entropy < EPSILON) {
        return 0.0;
    }
    
    return std::clamp(entropy / max_entropy, 0.0, 1.0);
}

// =============================================================================
// Spatial Uncertainty
// =============================================================================

double UncertaintyDecomposer::computeSpatialUncertainty(
    const SemanticPoint& point,
    const std::vector<SemanticPoint>& neighbors) const {
    
    if (neighbors.size() < static_cast<size_t>(config_.min_neighbors)) {
        // Not enough neighbors - high uncertainty
        return 1.0;
    }
    
    // Count neighbors with same class prediction
    int same_class_count = 0;
    for (const auto& neighbor : neighbors) {
        if (neighbor.semantic_class == point.semantic_class) {
            same_class_count++;
        }
    }
    
    // Consistency ratio
    double consistency = static_cast<double>(same_class_count) / 
                         static_cast<double>(neighbors.size());
    
    // Uncertainty is inverse of consistency
    return std::clamp(1.0 - consistency, 0.0, 1.0);
}

// =============================================================================
// Observation Uncertainty
// =============================================================================

double UncertaintyDecomposer::computeObservationUncertainty(
    const Vector3d& point,
    const Vector3d& sensor_origin,
    double local_density,
    const Vector3d& surface_normal) const {
    
    // Component 1: Range-based uncertainty
    // Increases with distance from sensor
    double range = (point - sensor_origin).norm();
    double u_range = config_.sigma_range * (range / sensor_model_.max_range);
    u_range = std::clamp(u_range, 0.0, 1.0);
    
    // Component 2: Density-based uncertainty
    // Increases in sparse regions
    double u_density = 0.0;
    if (local_density >= 0.0) {
        double density_ratio = local_density / config_.max_density;
        u_density = config_.sigma_density * (1.0 - std::min(density_ratio, 1.0));
    }
    
    // Component 3: Incidence angle uncertainty
    // Increases at grazing angles
    double u_angle = 0.0;
    if (surface_normal.norm() > 0.1) {
        Vector3d view_dir = (sensor_origin - point).normalized();
        Vector3d normal_normalized = surface_normal.normalized();
        double cos_angle = std::abs(view_dir.dot(normal_normalized));
        // Uncertainty increases as angle approaches 90 degrees (cos -> 0)
        u_angle = config_.sigma_angle * (1.0 - cos_angle);
    }
    
    // Combine components
    double uncertainty = u_range + u_density + u_angle;
    
    return std::clamp(uncertainty, 0.0, 1.0);
}

// =============================================================================
// Temporal Uncertainty
// =============================================================================

double UncertaintyDecomposer::computeTemporalUncertainty(
    const Vector3d& position,
    int current_class) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t hash = spatialHash(position);
    
    // Get or create history for this location
    auto it = temporal_history_.find(hash);
    if (it == temporal_history_.end()) {
        temporal_history_[hash] = TemporalHistory(DEFAULT_NUM_CLASSES);
        it = temporal_history_.find(hash);
    }
    
    TemporalHistory& history = it->second;
    
    // Update class counts
    if (current_class >= 0 && current_class < static_cast<int>(history.class_counts.size())) {
        history.class_counts[current_class]++;
    }
    history.total_observations++;
    
    // Need at least 2 observations for temporal analysis
    if (history.total_observations < 2) {
        return 0.5;  // Neutral uncertainty
    }
    
    // Find most frequent class
    int max_count = *std::max_element(history.class_counts.begin(), 
                                       history.class_counts.end());
    
    // Temporal consistency
    double consistency = static_cast<double>(max_count) / 
                         static_cast<double>(history.total_observations);
    
    // Uncertainty is inverse of consistency
    return std::clamp(1.0 - consistency, 0.0, 1.0);
}

// =============================================================================
// Combined Uncertainty
// =============================================================================

double UncertaintyDecomposer::computeTotalUncertainty(
    double u_sem, double u_spa, double u_obs, double u_temp) const {
    
    double total = config_.w_semantic * u_sem +
                   config_.w_spatial * u_spa +
                   config_.w_observation * u_obs +
                   config_.w_temporal * u_temp;
    
    return std::clamp(total, 0.0, 1.0);
}

UncertaintyDecomposition UncertaintyDecomposer::decompose(
    const SemanticPoint& point,
    const std::vector<SemanticPoint>& neighbors,
    const Vector3d& sensor_origin) {
    
    UncertaintyDecomposition result;
    
    // Use pre-computed semantic uncertainty if available, otherwise compute
    if (point.uncertainty_semantic >= 0.0 && point.uncertainty_semantic <= 1.0) {
        result.semantic = point.uncertainty_semantic;
    } else if (!point.evidence.empty()) {
        result.semantic = computeSemanticUncertainty(point.evidence, 
                                                      static_cast<int>(point.evidence.size()));
    } else if (!point.class_probabilities.empty()) {
        result.semantic = computeSemanticUncertaintyFromProbs(point.class_probabilities);
    } else {
        result.semantic = 1.0;
    }
    
    // Compute other components
    result.spatial = computeSpatialUncertainty(point, neighbors);
    result.observation = computeObservationUncertainty(point.position, sensor_origin);
    result.temporal = computeTemporalUncertainty(point.position, point.semantic_class);
    
    // Weighted combination
    result.total = computeTotalUncertainty(result.semantic, result.spatial,
                                            result.observation, result.temporal);
    
    return result;
}

// =============================================================================
// Batch Processing
// =============================================================================

void UncertaintyDecomposer::processPointCloud(
    std::vector<SemanticPoint>& points,
    const Vector3d& sensor_origin) {
    
    if (points.empty()) return;
    
    // For each point, find neighbors and compute uncertainty
    const size_t num_points = points.size();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        // Find neighbors within radius
        std::vector<SemanticPoint> neighbors;
        neighbors.reserve(50);
        
        for (size_t j = 0; j < num_points; ++j) {
            if (i == j) continue;
            
            double dist = (points[i].position - points[j].position).norm();
            if (dist < config_.spatial_radius) {
                neighbors.push_back(points[j]);
            }
        }
        
        // Compute local density
        double local_density = static_cast<double>(neighbors.size());
        
        // Decompose uncertainty
        auto decomp = decompose(points[i], neighbors, sensor_origin);
        
        // Update point
        points[i].uncertainty_semantic = decomp.semantic;
        points[i].uncertainty_spatial = decomp.spatial;
        points[i].uncertainty_observation = decomp.observation;
        points[i].uncertainty_temporal = decomp.temporal;
        points[i].uncertainty_total = decomp.total;
    }
}

void UncertaintyDecomposer::processPointCloudWithNeighbors(
    std::vector<SemanticPoint>& points,
    const std::vector<std::vector<int>>& neighbor_indices,
    const Vector3d& sensor_origin) {
    
    if (points.empty()) return;
    if (neighbor_indices.size() != points.size()) return;
    
    const size_t num_points = points.size();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        // Gather neighbors
        std::vector<SemanticPoint> neighbors;
        neighbors.reserve(neighbor_indices[i].size());
        
        for (int idx : neighbor_indices[i]) {
            if (idx >= 0 && idx < static_cast<int>(num_points) && idx != static_cast<int>(i)) {
                neighbors.push_back(points[idx]);
            }
        }
        
        // Decompose uncertainty
        auto decomp = decompose(points[i], neighbors, sensor_origin);
        
        // Update point
        points[i].uncertainty_semantic = decomp.semantic;
        points[i].uncertainty_spatial = decomp.spatial;
        points[i].uncertainty_observation = decomp.observation;
        points[i].uncertainty_temporal = decomp.temporal;
        points[i].uncertainty_total = decomp.total;
    }
}

// =============================================================================
// History Management
// =============================================================================

void UncertaintyDecomposer::resetTemporalHistory() {
    std::lock_guard<std::mutex> lock(mutex_);
    temporal_history_.clear();
}

void UncertaintyDecomposer::clearTemporalHistoryInRegion(
    const Vector3d& min_pt,
    const Vector3d& max_pt) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Iterate over all tracked positions
    // Note: This is inefficient for large histories; consider spatial indexing
    auto it = temporal_history_.begin();
    while (it != temporal_history_.end()) {
        // We don't have position stored directly, so we can't filter by region
        // In a production implementation, store position with history
        ++it;
    }
    
    // For now, just clear everything (simplified)
    // A proper implementation would use a spatial hash that can be queried by region
}

size_t UncertaintyDecomposer::getTemporalHistorySize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return temporal_history_.size();
}

// =============================================================================
// Configuration
// =============================================================================

void UncertaintyDecomposer::setWeights(double w_sem, double w_spa, 
                                        double w_obs, double w_temp) {
    config_.w_semantic = w_sem;
    config_.w_spatial = w_spa;
    config_.w_observation = w_obs;
    config_.w_temporal = w_temp;
    config_.normalizeWeights();
}

// =============================================================================
// Private Methods
// =============================================================================

size_t UncertaintyDecomposer::spatialHash(const Vector3d& position) const {
    // Discretize position to grid
    int x = static_cast<int>(std::floor(position.x() / config_.temporal_resolution));
    int y = static_cast<int>(std::floor(position.y() / config_.temporal_resolution));
    int z = static_cast<int>(std::floor(position.z() / config_.temporal_resolution));
    
    // Combine into hash
    size_t hash = 0;
    hash ^= std::hash<int>()(x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>()(y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>()(z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    
    return hash;
}

std::vector<int> UncertaintyDecomposer::findNeighbors(
    const std::vector<SemanticPoint>& points,
    int query_idx,
    double radius) const {
    
    std::vector<int> neighbors;
    const Vector3d& query_pos = points[query_idx].position;
    const double radius_sq = radius * radius;
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (static_cast<int>(i) == query_idx) continue;
        
        double dist_sq = (points[i].position - query_pos).squaredNorm();
        if (dist_sq < radius_sq) {
            neighbors.push_back(static_cast<int>(i));
        }
    }
    
    return neighbors;
}

double UncertaintyDecomposer::computeLocalDensity(
    const std::vector<SemanticPoint>& points,
    int query_idx,
    double radius) const {
    
    auto neighbors = findNeighbors(points, query_idx, radius);
    
    // Density = number of points / volume of sphere
    double volume = (4.0 / 3.0) * M_PI * radius * radius * radius;
    
    return static_cast<double>(neighbors.size()) / volume;
}

} // namespace hesfm
