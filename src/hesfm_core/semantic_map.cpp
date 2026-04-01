/**
 * @file semantic_map.cpp
 * @brief Implementation of semantic map with log-odds update
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include "hesfm/semantic_map.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>

namespace hesfm {

// =============================================================================
// Constructors
// =============================================================================

SemanticMap::SemanticMap()
    : config_(), traversable_classes_(DEFAULT_TRAVERSABLE_CLASSES) {}

SemanticMap::SemanticMap(const MapConfig& config)
    : config_(config), traversable_classes_(DEFAULT_TRAVERSABLE_CLASSES) {}

// =============================================================================
// Map Updates
// =============================================================================

void SemanticMap::update(const std::vector<GaussianPrimitive>& primitives,
                          const AdaptiveKernel& kernel) {
    
    if (primitives.empty()) return;
    
    // Compute max trace for length scale normalization
    double max_trace = 0.0;
    for (const auto& prim : primitives) {
        max_trace = std::max(max_trace, prim.covariance.trace());
    }
    
    update(primitives, kernel, max_trace);
}

void SemanticMap::update(const std::vector<GaussianPrimitive>& primitives,
                          const AdaptiveKernel& kernel,
                          double max_trace) {
    
    if (primitives.empty()) return;
    
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // For each primitive, update cells within its influence
    for (const auto& prim : primitives) {
        // Skip high uncertainty primitives
        if (prim.uncertainty > kernel.getConfig().uncertainty_threshold) {
            continue;
        }
        
        // Get adaptive length scale
        double length_scale = prim.computeAdaptiveLengthScale(
            kernel.getConfig().length_scale_min,
            kernel.getConfig().length_scale_max,
            max_trace);
        
        // Get bounding box of influence
        Vector3d min_bound, max_bound;
        kernel.getInfluenceBounds(prim, length_scale, min_bound, max_bound);
        
        // Clamp to map bounds
        min_bound.x() = std::max(min_bound.x(), config_.origin_x);
        min_bound.y() = std::max(min_bound.y(), config_.origin_y);
        min_bound.z() = std::max(min_bound.z(), config_.origin_z);
        max_bound.x() = std::min(max_bound.x(), config_.origin_x + config_.size_x);
        max_bound.y() = std::min(max_bound.y(), config_.origin_y + config_.size_y);
        max_bound.z() = std::min(max_bound.z(), config_.origin_z + config_.size_z);
        
        // Iterate over cells in bounding box
        for (double x = min_bound.x(); x <= max_bound.x(); x += config_.resolution) {
            for (double y = min_bound.y(); y <= max_bound.y(); y += config_.resolution) {
                for (double z = min_bound.z(); z <= max_bound.z(); z += config_.resolution) {
                    Vector3d cell_pos(x, y, z);
                    
                    // Compute kernel value
                    double k_val = kernel.compute(cell_pos, prim, max_trace);
                    if (k_val < EPSILON) continue;
                    
                    // Compute confidence weight
                    double w = kernel.computeConfidenceWeight(prim.uncertainty);
                    
                    // Get or create cell
                    MapCell& cell = getOrCreateCell(cell_pos);
                    
                    // Log-odds update for each class
                    for (int c = 0; c < config_.num_classes; ++c) {
                        double p = (c < static_cast<int>(prim.class_probabilities.size()))
                                   ? prim.class_probabilities[c]
                                   : config_.prior_prob;
                        
                        // Log-odds score
                        double log_odds_obs = probToLogOdds(p);
                        
                        // Update
                        cell.state.log_odds(c) += k_val * w * log_odds_obs;
                        cell.state.log_odds(c) = clampLogOdds(cell.state.log_odds(c));
                    }
                    
                    cell.state.observation_count++;
                    cell.state.last_update_time = prim.timestamp;

                    // --- Extended functional attributes (HESFM innovation #4) ---
                    int pred_class = cell.state.getPredictedClass();

                    // Update dynamic object status
                    cell.dynamic_status.update(pred_class, prim.timestamp);
                    cell.state.is_dynamic = cell.dynamic_status.is_dynamic;

                    // Update affordances from predicted class
                    cell.affordances.fromSemanticClass(pred_class);

                    // Propagate reachability from primitive
                    cell.state.reachability = std::max(
                        cell.state.reachability, prim.reachability);

                    // Update navigation cost incorporating affordances
                    if (cell.affordances.has(AffordanceType::TRAVERSABLE)) {
                        cell.nav_cost = 0;  // Free
                    } else if (cell.affordances.has(AffordanceType::AVOIDABLE)) {
                        cell.nav_cost = 100;  // Dynamic obstacle — always lethal
                    } else if (traversable_classes_.count(pred_class) > 0) {
                        cell.nav_cost = 0;  // Free
                    } else if (cell.state.getConfidence() > 0.5) {
                        cell.nav_cost = 100;  // Obstacle
                    }
                }
            }
        }
    }
    
    total_observations_ += primitives.size();

    // Enforce max_cells limit — prune lowest-confidence cells when exceeded
    if (config_.max_cells > 0 &&
        cells_.size() > static_cast<size_t>(config_.max_cells)) {
        // Find the confidence value at the pruning boundary
        std::vector<double> confidences;
        confidences.reserve(cells_.size());
        for (const auto& [h, c] : cells_) {
            confidences.push_back(c.state.getConfidence());
        }
        size_t excess = cells_.size() - config_.max_cells;
        std::nth_element(confidences.begin(),
                         confidences.begin() + static_cast<long>(excess),
                         confidences.end());
        double cutoff = confidences[excess];

        auto it = cells_.begin();
        size_t removed = 0;
        while (it != cells_.end() && removed < excess) {
            if (it->second.state.getConfidence() < cutoff) {
                it = cells_.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
    }
}

void SemanticMap::updateCell(const Vector3d& position,
                              const std::vector<double>& class_probs,
                              double weight) {
    
    if (!config_.isInBounds(position)) return;
    
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    MapCell& cell = getOrCreateCell(position);
    
    for (int c = 0; c < config_.num_classes && c < static_cast<int>(class_probs.size()); ++c) {
        double log_odds_obs = probToLogOdds(class_probs[c]);
        cell.state.log_odds(c) += weight * log_odds_obs;
        cell.state.log_odds(c) = clampLogOdds(cell.state.log_odds(c));
    }
    
    cell.state.observation_count++;
}

void SemanticMap::applyTemporalDecay(double decay_rate, double current_time, double max_age) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    for (auto& [hash, cell] : cells_) {
        double age = current_time - cell.state.last_update_time;
        
        if (age > max_age) {
            // Apply decay to log-odds (towards zero = uniform)
            cell.state.log_odds *= (1.0 - decay_rate);
        }
    }
}

// =============================================================================
// Map Queries
// =============================================================================

std::optional<SemanticState> SemanticMap::query(const Vector3d& position) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    size_t hash = positionToHash(position);
    auto it = cells_.find(hash);
    
    if (it == cells_.end()) {
        return std::nullopt;
    }
    
    return it->second.state;
}

int SemanticMap::getClass(const Vector3d& position) const {
    auto state = query(position);
    if (!state || state->observation_count == 0) {
        return -1;
    }
    return state->getPredictedClass();
}

VectorXd SemanticMap::getProbabilities(const Vector3d& position) const {
    auto state = query(position);
    if (!state) {
        return VectorXd::Constant(config_.num_classes, config_.prior_prob);
    }
    return state->getProbabilities();
}

double SemanticMap::getConfidence(const Vector3d& position) const {
    auto state = query(position);
    if (!state) return 0.0;
    return state->getConfidence();
}

double SemanticMap::getUncertainty(const Vector3d& position) const {
    return 1.0 - getConfidence(position);
}

bool SemanticMap::isTraversable(const Vector3d& position) const {
    int cls = getClass(position);
    if (cls < 0) return false;
    return traversable_classes_.count(cls) > 0;
}

bool SemanticMap::isObstacle(const Vector3d& position) const {
    auto cell = getCell(position);
    if (!cell) return false;
    return cell->nav_cost == 100;
}

std::optional<MapCell> SemanticMap::getCell(const Vector3d& position) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    size_t hash = positionToHash(position);
    auto it = cells_.find(hash);
    
    if (it == cells_.end()) {
        return std::nullopt;
    }
    
    return it->second;
}

std::vector<MapCell> SemanticMap::getOccupiedCells() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<MapCell> result;
    result.reserve(cells_.size());
    
    for (const auto& [hash, cell] : cells_) {
        if (cell.state.observation_count > 0) {
            result.push_back(cell);
        }
    }
    
    return result;
}

std::vector<MapCell> SemanticMap::getCellsInBBox(
    const Vector3d& min_pt,
    const Vector3d& max_pt) const {
    
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<MapCell> result;
    
    for (const auto& [hash, cell] : cells_) {
        const Vector3d& pos = cell.position;
        if (pos.x() >= min_pt.x() && pos.x() <= max_pt.x() &&
            pos.y() >= min_pt.y() && pos.y() <= max_pt.y() &&
            pos.z() >= min_pt.z() && pos.z() <= max_pt.z()) {
            result.push_back(cell);
        }
    }
    
    return result;
}

std::vector<MapCell> SemanticMap::getCellsByClass(int semantic_class,
                                                   double min_confidence) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<MapCell> result;
    
    for (const auto& [hash, cell] : cells_) {
        if (cell.state.getPredictedClass() == semantic_class &&
            cell.state.getConfidence() >= min_confidence) {
            result.push_back(cell);
        }
    }
    
    return result;
}

// =============================================================================
// Navigation Interface
// =============================================================================

std::vector<int8_t> SemanticMap::generateCostmap(
    double height_min,
    double height_max,
    int& width,
    int& height) const {
    
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    width = static_cast<int>(std::ceil(config_.size_x / config_.resolution));
    height = static_cast<int>(std::ceil(config_.size_y / config_.resolution));
    
    std::vector<int8_t> costmap(width * height, -1);  // Unknown
    
    for (const auto& [hash, cell] : cells_) {
        // Check height range
        if (cell.position.z() < height_min || cell.position.z() > height_max) {
            continue;
        }
        
        // Convert to grid coordinates
        int gx = static_cast<int>((cell.position.x() - config_.origin_x) / config_.resolution);
        int gy = static_cast<int>((cell.position.y() - config_.origin_y) / config_.resolution);
        
        if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;
        
        int idx = gy * width + gx;
        
        // Assign cost based on semantics
        int pred_class = cell.state.getPredictedClass();
        double confidence = cell.state.getConfidence();
        
        if (confidence < 0.3) {
            // Low confidence - keep as unknown unless already set
            if (costmap[idx] == -1) {
                costmap[idx] = -1;
            }
        } else if (traversable_classes_.count(pred_class) > 0) {
            // Traversable - free
            costmap[idx] = 0;
        } else {
            // Obstacle
            costmap[idx] = 100;
        }
    }
    
    return costmap;
}

std::vector<int8_t> SemanticMap::generateCostmap(
    const NavigationConfig& nav_config,
    int& width,
    int& height) const {
    
    return generateCostmap(nav_config.costmap_height_min,
                           nav_config.costmap_height_max,
                           width, height);
}

int8_t SemanticMap::getNavigationCost(const Vector3d& position) const {
    auto cell = getCell(position);
    if (!cell) return -1;
    return cell->nav_cost;
}

std::optional<Vector3d> SemanticMap::findNearestTraversable(
    const Vector3d& position,
    double max_distance) const {
    
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    double best_dist = std::numeric_limits<double>::max();
    std::optional<Vector3d> best_pos;
    
    for (const auto& [hash, cell] : cells_) {
        if (!cell.isTraversable(traversable_classes_)) continue;
        
        double dist = (cell.position - position).norm();
        if (dist < best_dist && dist < max_distance) {
            best_dist = dist;
            best_pos = cell.position;
        }
    }
    
    return best_pos;
}

// =============================================================================
// Map Management
// =============================================================================

void SemanticMap::reset() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cells_.clear();
    total_observations_ = 0;
}

void SemanticMap::resetRegion(const Vector3d& min_pt, const Vector3d& max_pt) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = cells_.begin();
    while (it != cells_.end()) {
        const Vector3d& pos = it->second.position;
        if (pos.x() >= min_pt.x() && pos.x() <= max_pt.x() &&
            pos.y() >= min_pt.y() && pos.y() <= max_pt.y() &&
            pos.z() >= min_pt.z() && pos.z() <= max_pt.z()) {
            it = cells_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t SemanticMap::pruneByConfidence(double min_confidence) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    size_t count = 0;
    auto it = cells_.begin();
    while (it != cells_.end()) {
        if (it->second.state.getConfidence() < min_confidence) {
            it = cells_.erase(it);
            count++;
        } else {
            ++it;
        }
    }
    
    return count;
}

size_t SemanticMap::pruneByObservationCount(int min_observations) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    size_t count = 0;
    auto it = cells_.begin();
    while (it != cells_.end()) {
        if (it->second.state.observation_count < min_observations) {
            it = cells_.erase(it);
            count++;
        } else {
            ++it;
        }
    }
    
    return count;
}

// =============================================================================
// Map I/O
// =============================================================================

bool SemanticMap::save(const std::string& filepath, const std::string& format) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    if (format == "yaml") {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        
        file << "# HESFM Semantic Map\n";
        file << "resolution: " << config_.resolution << "\n";
        file << "origin: [" << config_.origin_x << ", " 
             << config_.origin_y << ", " << config_.origin_z << "]\n";
        file << "num_classes: " << config_.num_classes << "\n";
        file << "num_cells: " << cells_.size() << "\n";
        file << "cells:\n";
        
        for (const auto& [hash, cell] : cells_) {
            file << "  - pos: [" << cell.position.x() << ", " 
                 << cell.position.y() << ", " << cell.position.z() << "]\n";
            file << "    class: " << cell.state.getPredictedClass() << "\n";
            file << "    confidence: " << cell.state.getConfidence() << "\n";
            file << "    observations: " << cell.state.observation_count << "\n";
        }
        
        file.close();
        return true;
    }
    
    return false;
}

bool SemanticMap::load(const std::string& filepath) {
    // Simplified loader - a full implementation would parse YAML
    return false;
}

std::vector<std::tuple<Vector3d, int, double>> SemanticMap::toPointCloud() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<std::tuple<Vector3d, int, double>> points;
    points.reserve(cells_.size());
    
    for (const auto& [hash, cell] : cells_) {
        points.emplace_back(
            cell.position,
            cell.state.getPredictedClass(),
            cell.state.getConfidence()
        );
    }
    
    return points;
}

// =============================================================================
// Statistics
// =============================================================================

double SemanticMap::getCoverage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // Total possible cells
    int nx, ny, nz;
    config_.getGridSize(nx, ny, nz);
    double total_cells = static_cast<double>(nx) * ny * nz;
    
    return static_cast<double>(cells_.size()) / total_cells;
}

double SemanticMap::getMeanConfidence() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    if (cells_.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& [hash, cell] : cells_) {
        sum += cell.state.getConfidence();
    }
    
    return sum / static_cast<double>(cells_.size());
}

double SemanticMap::getMeanUncertainty() const {
    return 1.0 - getMeanConfidence();
}

std::vector<int> SemanticMap::getClassDistribution() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<int> distribution(config_.num_classes, 0);
    
    for (const auto& [hash, cell] : cells_) {
        int cls = cell.state.getPredictedClass();
        if (cls >= 0 && cls < config_.num_classes) {
            distribution[cls]++;
        }
    }
    
    return distribution;
}

size_t SemanticMap::getMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    size_t cell_size = sizeof(MapCell) + config_.num_classes * sizeof(double);
    return cells_.size() * cell_size + sizeof(*this);
}

// =============================================================================
// Configuration
// =============================================================================

void SemanticMap::setConfig(const MapConfig& config) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    config_ = config;
}

// =============================================================================
// Private Methods
// =============================================================================

size_t SemanticMap::positionToHash(const Vector3d& position) const {
    int ix, iy, iz;
    if (!positionToIndex(position, ix, iy, iz)) {
        return 0;
    }
    
    size_t hash = 0;
    hash ^= std::hash<int>()(ix) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>()(iy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>()(iz) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    
    return hash;
}

bool SemanticMap::positionToIndex(const Vector3d& position,
                                   int& ix, int& iy, int& iz) const {
    if (!config_.isInBounds(position)) {
        return false;
    }
    
    ix = static_cast<int>(std::floor((position.x() - config_.origin_x) / config_.resolution));
    iy = static_cast<int>(std::floor((position.y() - config_.origin_y) / config_.resolution));
    iz = static_cast<int>(std::floor((position.z() - config_.origin_z) / config_.resolution));
    
    return true;
}

Vector3d SemanticMap::indexToPosition(int ix, int iy, int iz) const {
    return config_.indexToPosition(ix, iy, iz);
}

MapCell& SemanticMap::getOrCreateCell(const Vector3d& position) {
    size_t hash = positionToHash(position);
    
    auto it = cells_.find(hash);
    if (it == cells_.end()) {
        // Create new cell
        MapCell cell(config_.num_classes);
        int ix, iy, iz;
        positionToIndex(position, ix, iy, iz);
        cell.ix = ix;
        cell.iy = iy;
        cell.iz = iz;
        cell.position = indexToPosition(ix, iy, iz);
        
        cells_[hash] = cell;
        it = cells_.find(hash);
    }
    
    return it->second;
}

double SemanticMap::probToLogOdds(double prob) const {
    // Clamp to avoid log(0)
    prob = std::clamp(prob, EPSILON, 1.0 - EPSILON);
    return std::log(prob / config_.prior_prob);
}

double SemanticMap::logOddsToProb(double log_odds) const {
    return config_.prior_prob * std::exp(log_odds);
}

double SemanticMap::computeConfidenceWeight(double uncertainty, double entropy) const {
    double max_entropy = std::log(static_cast<double>(config_.num_classes));
    double w_unc = std::pow(1.0 - uncertainty, 2.0);
    double w_ent = std::pow(1.0 - entropy / max_entropy, 1.0);
    return w_unc * w_ent;
}

double SemanticMap::clampLogOdds(double log_odds) const {
    return std::clamp(log_odds, config_.log_odds_min, config_.log_odds_max);
}

} // namespace hesfm
