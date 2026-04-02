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
#include <ros/ros.h>

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

    // Advance the monotonic clock used for LRU eviction.
    temporal_clock_ += 1e-6;  // tiny increment per call, overwritten by real time below

    size_t hash = spatialHash(position);

    // Get or create history for this location
    auto it = temporal_history_.find(hash);
    if (it == temporal_history_.end()) {
        temporal_history_[hash] = TemporalHistory(DEFAULT_NUM_CLASSES);
        it = temporal_history_.find(hash);
    }

    TemporalHistory& history = it->second;
    history.last_access_time = temporal_clock_;

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

    // Periodically evict stale entries (every 10k accesses)
    static int access_count = 0;
    if (++access_count % 10000 == 0) {
        evictStaleHistory();
    }

    // Uncertainty is inverse of consistency
    return std::clamp(1.0 - consistency, 0.0, 1.0);
}

void UncertaintyDecomposer::evictStaleHistory() {
    // Called with mutex already held.
    const double cutoff = temporal_clock_ - TEMPORAL_HISTORY_MAX_AGE;

    // Phase 1: remove entries older than the sliding window
    auto it = temporal_history_.begin();
    while (it != temporal_history_.end()) {
        if (it->second.last_access_time < cutoff) {
            it = temporal_history_.erase(it);
        } else {
            ++it;
        }
    }

    // Phase 2: if still over capacity, remove oldest entries
    if (temporal_history_.size() > TEMPORAL_HISTORY_MAX_ENTRIES) {
        // Gather (time, hash) pairs and sort by time ascending
        std::vector<std::pair<double, size_t>> entries;
        entries.reserve(temporal_history_.size());
        for (const auto& [h, hist] : temporal_history_) {
            entries.emplace_back(hist.last_access_time, h);
        }
        std::sort(entries.begin(), entries.end());

        size_t to_remove = temporal_history_.size() - TEMPORAL_HISTORY_MAX_ENTRIES;
        for (size_t i = 0; i < to_remove && i < entries.size(); ++i) {
            temporal_history_.erase(entries[i].second);
        }
    }
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

    const size_t num_points = points.size();
    const double inv_radius = 1.0 / config_.spatial_radius;

    // ── Phase 0: Build voxel grid for O(N) neighbor queries ──────────────────
    // Voxel side = spatial_radius so all neighbors within radius lie in ±1 voxel.
    auto voxelHash = [](int ix, int iy, int iz) -> size_t {
        size_t h = 0;
        h ^= std::hash<int>()(ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    };

    std::unordered_map<size_t, std::vector<int>> voxel_grid;
    voxel_grid.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        int ix = static_cast<int>(std::floor(points[i].position.x() * inv_radius));
        int iy = static_cast<int>(std::floor(points[i].position.y() * inv_radius));
        int iz = static_cast<int>(std::floor(points[i].position.z() * inv_radius));
        voxel_grid[voxelHash(ix, iy, iz)].push_back(static_cast<int>(i));
    }

    // ── Phase 1 (parallel): U_sem, U_spa, U_obs — no mutex ──────────────────
    struct PartialResult { double u_sem, u_spa, u_obs; };
    std::vector<PartialResult> partial(num_points);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < num_points; ++i) {
        const SemanticPoint& pt = points[i];

        // Semantic uncertainty (use pre-computed value from segmentation if valid)
        double u_sem;
        if (pt.uncertainty_semantic >= 0.0 && pt.uncertainty_semantic <= 1.0) {
            u_sem = pt.uncertainty_semantic;
        } else if (!pt.evidence.empty()) {
            u_sem = computeSemanticUncertainty(pt.evidence,
                                               static_cast<int>(pt.evidence.size()));
        } else if (!pt.class_probabilities.empty()) {
            u_sem = computeSemanticUncertaintyFromProbs(pt.class_probabilities);
        } else {
            u_sem = 1.0;
        }

        // Spatial uncertainty via voxel grid
        int ix = static_cast<int>(std::floor(pt.position.x() * inv_radius));
        int iy = static_cast<int>(std::floor(pt.position.y() * inv_radius));
        int iz = static_cast<int>(std::floor(pt.position.z() * inv_radius));

        std::vector<SemanticPoint> neighbors;
        neighbors.reserve(32);
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    auto it = voxel_grid.find(voxelHash(ix+dx, iy+dy, iz+dz));
                    if (it == voxel_grid.end()) continue;
                    for (int j : it->second) {
                        if (j == static_cast<int>(i)) continue;
                        if ((pt.position - points[j].position).norm()
                                < config_.spatial_radius) {
                            neighbors.push_back(points[j]);
                        }
                    }
                }
            }
        }

        double u_spa = computeSpatialUncertainty(pt, neighbors);
        double u_obs = computeObservationUncertainty(pt.position, sensor_origin);

        partial[i] = {u_sem, u_spa, u_obs};
    }

    // ── Phase 2 (sequential): U_temp + total write-back ──────────────────────
    // computeTemporalUncertainty() updates shared state — keep sequential.
    double sum_sem = 0, sum_spa = 0, sum_obs = 0, sum_temp = 0, sum_total = 0;

    for (size_t i = 0; i < num_points; ++i) {
        double u_temp = computeTemporalUncertainty(points[i].position,
                                                    points[i].semantic_class);
        double u_total = computeTotalUncertainty(partial[i].u_sem, partial[i].u_spa,
                                                  partial[i].u_obs, u_temp);

        points[i].uncertainty_semantic    = partial[i].u_sem;
        points[i].uncertainty_spatial     = partial[i].u_spa;
        points[i].uncertainty_observation = partial[i].u_obs;
        points[i].uncertainty_temporal    = u_temp;
        points[i].uncertainty_total       = u_total;

        sum_sem   += partial[i].u_sem;
        sum_spa   += partial[i].u_spa;
        sum_obs   += partial[i].u_obs;
        sum_temp  += u_temp;
        sum_total += u_total;
    }

    // ── Diagnostics: log per-component means every 30 clouds ─────────────────
    static int call_count = 0;
    if (++call_count % 30 == 1) {
        double n = static_cast<double>(num_points);
        ROS_INFO("[Uncertainty] n=%zu  sem=%.3f  spa=%.3f  obs=%.3f  temp=%.3f  total=%.3f",
                 num_points,
                 sum_sem/n, sum_spa/n, sum_obs/n, sum_temp/n, sum_total/n);
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

    // Clear all hashes that correspond to positions within the bounding box.
    // We enumerate the spatial grid cells that cover [min_pt, max_pt] and erase
    // matching entries. This is exact for the grid-based hash scheme used by
    // spatialHash() (floor(pos / temporal_resolution)).
    const double res = config_.temporal_resolution;
    const int x_min = static_cast<int>(std::floor(min_pt.x() / res));
    const int x_max = static_cast<int>(std::floor(max_pt.x() / res));
    const int y_min = static_cast<int>(std::floor(min_pt.y() / res));
    const int y_max = static_cast<int>(std::floor(max_pt.y() / res));
    const int z_min = static_cast<int>(std::floor(min_pt.z() / res));
    const int z_max = static_cast<int>(std::floor(max_pt.z() / res));

    for (int ix = x_min; ix <= x_max; ++ix) {
        for (int iy = y_min; iy <= y_max; ++iy) {
            for (int iz = z_min; iz <= z_max; ++iz) {
                size_t hash = 0;
                hash ^= std::hash<int>()(ix) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                hash ^= std::hash<int>()(iy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                hash ^= std::hash<int>()(iz) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                temporal_history_.erase(hash);
            }
        }
    }
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

    // Voxel-grid accelerated neighbor search — O(N) build + O(k) per query
    // instead of the original O(N) brute-force per query.
    const double inv_r = 1.0 / radius;
    auto vHash = [](int ix, int iy, int iz) -> size_t {
        size_t h = 0;
        h ^= std::hash<int>()(ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    };

    // Build grid (amortised over all queries via static thread-local cache
    // would be better, but for correctness we rebuild here — this function
    // is only used as a fallback; processPointCloud() already has its own
    // voxel grid in Phase 0).
    std::unordered_map<size_t, std::vector<int>> grid;
    grid.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        int ix = static_cast<int>(std::floor(points[i].position.x() * inv_r));
        int iy = static_cast<int>(std::floor(points[i].position.y() * inv_r));
        int iz = static_cast<int>(std::floor(points[i].position.z() * inv_r));
        grid[vHash(ix, iy, iz)].push_back(static_cast<int>(i));
    }

    const Vector3d& qp = points[query_idx].position;
    int qx = static_cast<int>(std::floor(qp.x() * inv_r));
    int qy = static_cast<int>(std::floor(qp.y() * inv_r));
    int qz = static_cast<int>(std::floor(qp.z() * inv_r));
    const double radius_sq = radius * radius;

    std::vector<int> neighbors;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                auto it = grid.find(vHash(qx + dx, qy + dy, qz + dz));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j == query_idx) continue;
                    if ((points[j].position - qp).squaredNorm() < radius_sq) {
                        neighbors.push_back(j);
                    }
                }
            }
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
