/**
 * @file gaussian_primitives.cpp
 * @brief Implementation of Gaussian primitive construction
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include "hesfm/gaussian_primitives.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace hesfm {

// =============================================================================
// Constructors
// =============================================================================

GaussianPrimitiveBuilder::GaussianPrimitiveBuilder()
    : config_(), rng_(std::random_device{}()) {}

GaussianPrimitiveBuilder::GaussianPrimitiveBuilder(const PrimitiveConfig& config)
    : config_(config), rng_(std::random_device{}()) {}

// =============================================================================
// Main Building Interface
// =============================================================================

std::vector<GaussianPrimitive> GaussianPrimitiveBuilder::buildPrimitives(
    const std::vector<SemanticPoint>& points) {
    
    if (points.size() < static_cast<size_t>(config_.min_points_per_primitive)) {
        return {};
    }
    
    // Determine number of clusters
    int k = std::min(config_.target_primitives,
                     static_cast<int>(points.size() / config_.min_points_per_primitive));
    k = std::max(k, 1);
    
    // Uncertainty-weighted K-means clustering
    std::vector<Vector3d> centroids;
    std::vector<int> assignments = uncertaintyWeightedKMeans(points, k, centroids);
    
    // Build primitive for each cluster
    std::vector<GaussianPrimitive> primitives;
    primitives.reserve(k);
    
    for (int c = 0; c < k; ++c) {
        // Gather indices of points in this cluster
        std::vector<int> cluster_indices;
        for (size_t i = 0; i < assignments.size(); ++i) {
            if (assignments[i] == c) {
                cluster_indices.push_back(static_cast<int>(i));
            }
        }
        
        // Skip if too few points
        if (cluster_indices.size() < static_cast<size_t>(config_.min_points_per_primitive)) {
            continue;
        }
        
        // Compute primitive
        GaussianPrimitive prim = computePrimitive(points, cluster_indices);
        
        // Filter by conflict
        if (prim.conflict < config_.conflict_threshold) {
            primitives.push_back(prim);
        }
    }
    
    return primitives;
}

std::vector<GaussianPrimitive> GaussianPrimitiveBuilder::buildPrimitivesAuto(
    const std::vector<SemanticPoint>& points,
    int min_primitives,
    int max_primitives) {
    
    // Estimate optimal number based on point distribution
    // Use rule of thumb: sqrt(N/2) clusters for N points
    int estimated = static_cast<int>(std::sqrt(points.size() / 2.0));
    int k = std::clamp(estimated, min_primitives, max_primitives);
    
    // Temporarily set target
    int original_target = config_.target_primitives;
    config_.target_primitives = k;
    
    auto primitives = buildPrimitives(points);
    
    config_.target_primitives = original_target;
    
    return primitives;
}

// =============================================================================
// Incremental Updates
// =============================================================================

std::vector<GaussianPrimitive> GaussianPrimitiveBuilder::updatePrimitives(
    const std::vector<GaussianPrimitive>& primitives,
    const std::vector<SemanticPoint>& new_points,
    double max_distance) {
    
    std::vector<GaussianPrimitive> updated = primitives;
    std::vector<bool> point_assigned(new_points.size(), false);
    
    // Associate new points with existing primitives
    for (size_t i = 0; i < new_points.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_prim = -1;
        
        for (size_t j = 0; j < updated.size(); ++j) {
            double dist = (new_points[i].position - updated[j].centroid).norm();
            if (dist < min_dist && dist < max_distance) {
                min_dist = dist;
                best_prim = static_cast<int>(j);
            }
        }
        
        if (best_prim >= 0) {
            // Update primitive with new point
            auto& prim = updated[best_prim];
            double weight = new_points[i].getWeight();
            double total_weight = prim.total_weight + weight;
            
            // Update centroid (weighted running average)
            prim.centroid = (prim.centroid * prim.total_weight + 
                            new_points[i].position * weight) / total_weight;
            
            // Update class probabilities via DST fusion
            double conflict;
            prim.class_probabilities = dstFusion(
                prim.class_probabilities,
                new_points[i].class_probabilities,
                prim.uncertainty,
                new_points[i].uncertainty_total,
                conflict);
            prim.conflict = std::max(prim.conflict, conflict);
            
            prim.point_count++;
            prim.total_weight = total_weight;
            prim.uncertainty = (prim.uncertainty * (prim.point_count - 1) + 
                               new_points[i].uncertainty_total) / prim.point_count;
            
            point_assigned[i] = true;
        }
    }
    
    // Create new primitives for unassigned points
    std::vector<SemanticPoint> unassigned;
    for (size_t i = 0; i < new_points.size(); ++i) {
        if (!point_assigned[i]) {
            unassigned.push_back(new_points[i]);
        }
    }
    
    if (unassigned.size() >= static_cast<size_t>(config_.min_points_per_primitive)) {
        auto new_prims = buildPrimitives(unassigned);
        updated.insert(updated.end(), new_prims.begin(), new_prims.end());
    }
    
    return updated;
}

GaussianPrimitive GaussianPrimitiveBuilder::mergePrimitives(
    const GaussianPrimitive& prim1,
    const GaussianPrimitive& prim2) {
    
    GaussianPrimitive merged;
    merged.id = getNextPrimitiveId();
    
    double total_weight = prim1.total_weight + prim2.total_weight;
    
    // Merge centroid
    merged.centroid = (prim1.centroid * prim1.total_weight +
                       prim2.centroid * prim2.total_weight) / total_weight;
    
    // Merge covariance using parallel axis theorem
    Vector3d d1 = prim1.centroid - merged.centroid;
    Vector3d d2 = prim2.centroid - merged.centroid;
    
    double w1 = prim1.total_weight / total_weight;
    double w2 = prim2.total_weight / total_weight;
    
    merged.covariance = w1 * (prim1.covariance + d1 * d1.transpose()) +
                        w2 * (prim2.covariance + d2 * d2.transpose());
    
    // Merge semantics via DST
    double conflict;
    merged.class_probabilities = dstFusion(
        prim1.class_probabilities,
        prim2.class_probabilities,
        prim1.uncertainty,
        prim2.uncertainty,
        conflict);
    merged.conflict = conflict;
    
    // Update predicted class
    auto max_it = std::max_element(merged.class_probabilities.begin(),
                                    merged.class_probabilities.end());
    merged.semantic_class = static_cast<int>(
        std::distance(merged.class_probabilities.begin(), max_it));
    
    merged.point_count = prim1.point_count + prim2.point_count;
    merged.total_weight = total_weight;
    merged.uncertainty = (prim1.uncertainty * prim1.point_count +
                          prim2.uncertainty * prim2.point_count) / merged.point_count;
    
    return merged;
}

std::vector<GaussianPrimitive> GaussianPrimitiveBuilder::splitPrimitive(
    const GaussianPrimitive& primitive,
    const std::vector<SemanticPoint>& points,
    int num_splits) {
    
    if (points.size() < static_cast<size_t>(num_splits * config_.min_points_per_primitive)) {
        return {primitive};
    }
    
    // Temporarily modify config
    int original_target = config_.target_primitives;
    config_.target_primitives = num_splits;
    
    auto split_primitives = buildPrimitives(points);
    
    config_.target_primitives = original_target;
    
    if (split_primitives.empty()) {
        return {primitive};
    }
    
    return split_primitives;
}

// =============================================================================
// Dempster-Shafer Fusion
// =============================================================================

std::vector<double> GaussianPrimitiveBuilder::dstFusion(
    const std::vector<double>& belief1,
    const std::vector<double>& belief2,
    double uncertainty1,
    double uncertainty2,
    double& conflict) {
    
    // Handle empty inputs
    if (belief1.empty()) return belief2;
    if (belief2.empty()) return belief1;
    
    size_t n = std::max(belief1.size(), belief2.size());
    std::vector<double> result(n, 0.0);
    
    // Normalize inputs if needed
    auto normalize = [](const std::vector<double>& v) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        std::vector<double> normalized(v.size());
        if (sum > EPSILON) {
            for (size_t i = 0; i < v.size(); ++i) {
                normalized[i] = v[i] / sum;
            }
        }
        return normalized;
    };
    
    std::vector<double> m1 = normalize(belief1);
    std::vector<double> m2 = normalize(belief2);
    
    // Adjust for uncertainty (mass on frame Theta)
    // Higher uncertainty means more mass on "don't know"
    double m1_theta = uncertainty1;
    double m2_theta = uncertainty2;
    
    // Scale down specific beliefs
    for (auto& m : m1) m *= (1.0 - m1_theta);
    for (auto& m : m2) m *= (1.0 - m2_theta);
    
    // Compute conflict: K = sum over disjoint pairs
    conflict = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                double b1 = (i < m1.size()) ? m1[i] : 0.0;
                double b2 = (j < m2.size()) ? m2[j] : 0.0;
                conflict += b1 * b2;
            }
        }
    }
    
    // Normalization factor
    double norm = 1.0 - conflict;
    if (norm < EPSILON) {
        norm = EPSILON;
        conflict = 1.0 - EPSILON;
    }
    
    // Dempster's combination rule
    for (size_t c = 0; c < n; ++c) {
        double b1 = (c < m1.size()) ? m1[c] : 0.0;
        double b2 = (c < m2.size()) ? m2[c] : 0.0;
        
        // Combined mass = (m1(c)*m2(c) + m1(c)*m2(Theta) + m1(Theta)*m2(c)) / (1-K)
        result[c] = (b1 * b2 + b1 * m2_theta + m1_theta * b2) / norm;
    }
    
    // Ensure normalization
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    if (sum > EPSILON) {
        for (auto& r : result) r /= sum;
    }
    
    return result;
}

std::vector<double> GaussianPrimitiveBuilder::dstFusionMultiple(
    const std::vector<std::vector<double>>& beliefs,
    const std::vector<double>& uncertainties,
    double& total_conflict) {
    
    if (beliefs.empty()) return {};
    if (beliefs.size() == 1) {
        total_conflict = 0.0;
        return beliefs[0];
    }
    
    // Sequential pairwise fusion
    std::vector<double> result = beliefs[0];
    double current_uncertainty = uncertainties.empty() ? 0.5 : uncertainties[0];
    total_conflict = 0.0;
    
    for (size_t i = 1; i < beliefs.size(); ++i) {
        double pairwise_conflict;
        double next_uncertainty = (i < uncertainties.size()) ? uncertainties[i] : 0.5;
        
        result = dstFusion(result, beliefs[i], 
                          current_uncertainty, next_uncertainty,
                          pairwise_conflict);
        
        total_conflict = std::max(total_conflict, pairwise_conflict);
        
        // Update uncertainty for next iteration (average)
        current_uncertainty = (current_uncertainty + next_uncertainty) / 2.0;
    }
    
    return result;
}

// =============================================================================
// Clustering Methods
// =============================================================================

std::vector<Vector3d> GaussianPrimitiveBuilder::kmeansppInit(
    const std::vector<SemanticPoint>& points,
    int k) {
    
    std::vector<Vector3d> centroids;
    centroids.reserve(k);
    
    // Compute weights (inverse uncertainty)
    std::vector<double> weights(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        weights[i] = 1.0 / (points[i].uncertainty_total + 0.1);
    }
    
    // First centroid: weighted random selection
    std::discrete_distribution<size_t> init_dist(weights.begin(), weights.end());
    centroids.push_back(points[init_dist(rng_)].position);
    
    // Remaining centroids: K-means++ selection
    for (int c = 1; c < k; ++c) {
        std::vector<double> min_dists(points.size());
        
        for (size_t i = 0; i < points.size(); ++i) {
            double min_d = std::numeric_limits<double>::max();
            for (const auto& centroid : centroids) {
                double d = (points[i].position - centroid).squaredNorm();
                // Weight by uncertainty
                d *= (1.0 + config_.uncertainty_weight_lambda * points[i].uncertainty_total);
                min_d = std::min(min_d, d);
            }
            min_dists[i] = min_d;
        }
        
        std::discrete_distribution<size_t> dist(min_dists.begin(), min_dists.end());
        centroids.push_back(points[dist(rng_)].position);
    }
    
    return centroids;
}

std::vector<int> GaussianPrimitiveBuilder::uncertaintyWeightedKMeans(
    const std::vector<SemanticPoint>& points,
    int k,
    std::vector<Vector3d>& centroids) {
    
    // Initialize centroids with K-means++
    centroids = kmeansppInit(points, k);
    
    std::vector<int> assignments(points.size(), 0);
    
    // Iterative refinement
    for (int iter = 0; iter < config_.kmeans_max_iter; ++iter) {
        // Assignment step
        for (size_t i = 0; i < points.size(); ++i) {
            double min_d = std::numeric_limits<double>::max();
            int best_c = 0;
            
            for (int c = 0; c < k; ++c) {
                double d = (points[i].position - centroids[c]).squaredNorm();
                // Weight by uncertainty
                d *= (1.0 + config_.uncertainty_weight_lambda * points[i].uncertainty_total);
                
                if (d < min_d) {
                    min_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }
        
        // Update step (weighted centroids)
        std::vector<Vector3d> new_centroids(k, Vector3d::Zero());
        std::vector<double> total_weights(k, 0.0);
        
        for (size_t i = 0; i < points.size(); ++i) {
            int c = assignments[i];
            double w = 1.0 / (points[i].uncertainty_total + 0.1);
            
            new_centroids[c] += w * points[i].position;
            total_weights[c] += w;
        }
        
        double max_shift = 0.0;
        for (int c = 0; c < k; ++c) {
            if (total_weights[c] > EPSILON) {
                new_centroids[c] /= total_weights[c];
                max_shift = std::max(max_shift,
                                    (new_centroids[c] - centroids[c]).norm());
                centroids[c] = new_centroids[c];
            } else {
                // Empty cluster: re-seed from the point furthest from its
                // assigned centroid (highest quantization error).
                double best_d = 0.0;
                size_t best_i = 0;
                for (size_t i = 0; i < points.size(); ++i) {
                    double d = (points[i].position - centroids[assignments[i]]).squaredNorm();
                    if (d > best_d) { best_d = d; best_i = i; }
                }
                centroids[c] = points[best_i].position;
                max_shift = std::numeric_limits<double>::max(); // force re-assign
            }
        }

        // Check convergence
        if (max_shift < config_.kmeans_tolerance) {
            break;
        }
    }
    
    return assignments;
}

// =============================================================================
// Utility Methods
// =============================================================================

double GaussianPrimitiveBuilder::computeMaxTrace(
    const std::vector<GaussianPrimitive>& primitives) const {
    
    double max_trace = 0.0;
    for (const auto& prim : primitives) {
        max_trace = std::max(max_trace, prim.covariance.trace());
    }
    return max_trace;
}

std::vector<GaussianPrimitive> GaussianPrimitiveBuilder::filterByConflict(
    const std::vector<GaussianPrimitive>& primitives,
    double threshold) const {
    
    std::vector<GaussianPrimitive> filtered;
    filtered.reserve(primitives.size());
    
    for (const auto& prim : primitives) {
        if (prim.conflict < threshold) {
            filtered.push_back(prim);
        }
    }
    
    return filtered;
}

void GaussianPrimitiveBuilder::computeStatistics(
    const std::vector<GaussianPrimitive>& primitives,
    double& mean_points,
    double& mean_uncertainty,
    double& mean_conflict) const {
    
    if (primitives.empty()) {
        mean_points = mean_uncertainty = mean_conflict = 0.0;
        return;
    }
    
    double sum_points = 0.0, sum_unc = 0.0, sum_conf = 0.0;
    
    for (const auto& prim : primitives) {
        sum_points += prim.point_count;
        sum_unc += prim.uncertainty;
        sum_conf += prim.conflict;
    }
    
    double n = static_cast<double>(primitives.size());
    mean_points = sum_points / n;
    mean_uncertainty = sum_unc / n;
    mean_conflict = sum_conf / n;
}

// =============================================================================
// Private Methods
// =============================================================================

GaussianPrimitive GaussianPrimitiveBuilder::computePrimitive(
    const std::vector<SemanticPoint>& points,
    const std::vector<int>& indices) {
    
    GaussianPrimitive prim;
    prim.id = getNextPrimitiveId();
    prim.point_count = static_cast<int>(indices.size());
    prim.class_probabilities.resize(config_.num_classes, 0.0);
    
    // Compute weights
    auto weights = computePointWeights(points, indices);
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    prim.total_weight = total_weight;
    
    // Normalize weights
    for (auto& w : weights) w /= total_weight;
    
    // Weighted centroid
    prim.centroid = computeWeightedCentroid(points, indices, weights);
    
    // Weighted covariance
    prim.covariance = computeWeightedCovariance(points, indices, weights, prim.centroid);
    
    // Add regularization to prevent singularity
    prim.covariance += config_.regularization * Matrix3d::Identity();
    
    // DST fusion for semantics
    std::vector<std::vector<double>> all_probs;
    std::vector<double> all_uncertainties;
    all_probs.reserve(indices.size());
    all_uncertainties.reserve(indices.size());
    
    for (int idx : indices) {
        all_probs.push_back(points[idx].class_probabilities);
        all_uncertainties.push_back(points[idx].uncertainty_semantic);
    }
    
    prim.class_probabilities = dstFusionMultiple(all_probs, all_uncertainties, prim.conflict);
    
    // Predicted class
    if (!prim.class_probabilities.empty()) {
        auto max_it = std::max_element(prim.class_probabilities.begin(),
                                        prim.class_probabilities.end());
        prim.semantic_class = static_cast<int>(
            std::distance(prim.class_probabilities.begin(), max_it));
    }
    
    // Average uncertainty
    prim.uncertainty = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        prim.uncertainty += weights[i] * points[indices[i]].uncertainty_total;
    }
    
    return prim;
}

Vector3d GaussianPrimitiveBuilder::computeWeightedCentroid(
    const std::vector<SemanticPoint>& points,
    const std::vector<int>& indices,
    const std::vector<double>& weights) {
    
    Vector3d centroid = Vector3d::Zero();
    
    for (size_t i = 0; i < indices.size(); ++i) {
        centroid += weights[i] * points[indices[i]].position;
    }
    
    return centroid;
}

Matrix3d GaussianPrimitiveBuilder::computeWeightedCovariance(
    const std::vector<SemanticPoint>& points,
    const std::vector<int>& indices,
    const std::vector<double>& weights,
    const Vector3d& centroid) {
    
    Matrix3d covariance = Matrix3d::Zero();
    
    for (size_t i = 0; i < indices.size(); ++i) {
        Vector3d diff = points[indices[i]].position - centroid;
        covariance += weights[i] * diff * diff.transpose();
    }
    
    return covariance;
}

std::vector<double> GaussianPrimitiveBuilder::computePointWeights(
    const std::vector<SemanticPoint>& points,
    const std::vector<int>& indices) {
    
    std::vector<double> weights(indices.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        weights[i] = points[indices[i]].getWeight(config_.uncertainty_weight_lambda);
    }
    
    return weights;
}

} // namespace hesfm
