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
#include <sstream>
#include <unordered_set>

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

    // Track modified cells for deferred extended-attribute update
    std::unordered_set<size_t> modified_hashes;

    // Direct hash from grid indices (avoids position→index→hash round-trip)
    auto hashFromIndex = [](int ix, int iy, int iz) -> size_t {
        size_t h = 0;
        h ^= std::hash<int>()(ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    };

    const double two_pi = 2.0 * M_PI;
    const double inv_two_pi = 1.0 / two_pi;
    const double reach_lambda = kernel.getConfig().reachability_lambda;
    const auto& trav_classes = kernel.getConfig().traversable_classes;
    const double lo_min = config_.log_odds_min;
    const double lo_max = config_.log_odds_max;

    for (const auto& prim : primitives) {
        // Skip high uncertainty primitives
        if (prim.uncertainty > kernel.getConfig().uncertainty_threshold) {
            continue;
        }

        // ==== PRECOMPUTE per-primitive (ONCE, not per-cell) ====

        // Use uncertainty-adaptive length scale (consistent for bounding box AND kernel)
        double length_scale = kernel.computeUncertaintyAdaptiveLengthScale(
            prim.covariance, max_trace, prim.uncertainty);

        // Inverse covariance — precomputed ONCE instead of SVD per cell
        Matrix3d cov_inv = (prim.covariance + 0.001 * Matrix3d::Identity()).inverse();

        // Uncertainty-related weights (constant per primitive)
        double k_unc = kernel.uncertaintyKernel(prim.uncertainty);
        double k_dyn = prim.is_dynamic ? 0.5 : 1.0;
        double w = kernel.computeConfidenceWeight(prim.uncertainty);
        double kw = k_unc * k_dyn * w;  // combined per-primitive multiplier

        bool is_traversable = trav_classes.count(prim.semantic_class) > 0;

        // Precompute log-odds for all classes (avoid 37× probToLogOdds per cell)
        VectorXd log_odds_obs(config_.num_classes);
        for (int c = 0; c < config_.num_classes; ++c) {
            double p = (c < static_cast<int>(prim.class_probabilities.size()))
                       ? prim.class_probabilities[c]
                       : config_.prior_prob;
            log_odds_obs(c) = probToLogOdds(p);
        }

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

        // Grid index ranges
        int ix_min = static_cast<int>(std::floor((min_bound.x() - config_.origin_x) / config_.resolution));
        int iy_min = static_cast<int>(std::floor((min_bound.y() - config_.origin_y) / config_.resolution));
        int iz_min = static_cast<int>(std::floor((min_bound.z() - config_.origin_z) / config_.resolution));
        int ix_max = static_cast<int>(std::floor((max_bound.x() - config_.origin_x) / config_.resolution));
        int iy_max = static_cast<int>(std::floor((max_bound.y() - config_.origin_y) / config_.resolution));
        int iz_max = static_cast<int>(std::floor((max_bound.z() - config_.origin_z) / config_.resolution));

        // ==== CELL LOOP (optimized: inline kernel, no per-cell SVD) ====
        for (int ix = ix_min; ix <= ix_max; ++ix) {
            double x = config_.origin_x + ix * config_.resolution;
            double dx = x - prim.centroid.x();
            for (int iy = iy_min; iy <= iy_max; ++iy) {
                double y = config_.origin_y + iy * config_.resolution;
                double dy = y - prim.centroid.y();
                for (int iz = iz_min; iz <= iz_max; ++iz) {
                    double z = config_.origin_z + iz * config_.resolution;
                    double dz = z - prim.centroid.z();

                    // Inline Mahalanobis distance (uses precomputed cov_inv)
                    double d_sq = dx * (cov_inv(0,0)*dx + cov_inv(0,1)*dy + cov_inv(0,2)*dz)
                                + dy * (cov_inv(1,0)*dx + cov_inv(1,1)*dy + cov_inv(1,2)*dz)
                                + dz * (cov_inv(2,0)*dx + cov_inv(2,1)*dy + cov_inv(2,2)*dz);
                    double d_M = std::sqrt(std::max(0.0, d_sq));

                    // Compact support: skip cells outside length scale
                    if (d_M >= length_scale) continue;

                    // Inline Wendland C2 sparse kernel
                    double d_norm = d_M / length_scale;
                    double k_geo = ((2.0 + std::cos(two_pi * d_norm)) * (1.0 - d_norm) / 3.0
                                   + std::sin(two_pi * d_norm) * inv_two_pi);
                    if (k_geo < EPSILON) continue;

                    // Reachability kernel
                    double k_reach = is_traversable ? 1.0
                                   : std::exp(-reach_lambda * std::sqrt(dx*dx + dy*dy + dz*dz));

                    // Combined scale
                    double scale = k_geo * k_reach * kw;

                    // Get or create cell (direct hash, single lookup via try_emplace)
                    size_t hash = hashFromIndex(ix, iy, iz);
                    auto [it, inserted] = cells_.try_emplace(hash, config_.num_classes);
                    if (inserted) {
                        it->second.ix = ix;
                        it->second.iy = iy;
                        it->second.iz = iz;
                        it->second.position = config_.indexToPosition(ix, iy, iz);
                    }
                    MapCell& cell = it->second;

                    // Vectorized log-odds update (Eigen vector op)
                    cell.state.log_odds.noalias() += scale * log_odds_obs;
                    cell.state.log_odds = cell.state.log_odds.cwiseMax(lo_min).cwiseMin(lo_max);

                    cell.state.observation_count++;
                    cell.state.last_update_time = prim.timestamp;
                    cell.state.reachability = std::max(cell.state.reachability, prim.reachability);

                    modified_hashes.insert(hash);
                }
            }
        }
    }

    // ==== DEFERRED: extended attributes (single pass, once per cell) ====
    for (size_t hash : modified_hashes) {
        auto it = cells_.find(hash);
        if (it == cells_.end()) continue;
        MapCell& cell = it->second;

        int pred_class = cell.state.getPredictedClass();

        cell.dynamic_status.update(pred_class, cell.state.last_update_time);
        cell.state.is_dynamic = cell.dynamic_status.is_dynamic;
        cell.affordances.fromSemanticClass(pred_class);

        if (cell.affordances.has(AffordanceType::TRAVERSABLE)) {
            cell.nav_cost = 0;
        } else if (cell.affordances.has(AffordanceType::AVOIDABLE)) {
            cell.nav_cost = 100;
        } else if (traversable_classes_.count(pred_class) > 0) {
            cell.nav_cost = 0;
        } else if (cell.state.getConfidence() > 0.5) {
            cell.nav_cost = 100;
        }
    }

    total_observations_ += primitives.size();

    // Enforce max_cells limit — prune lowest-confidence cells when exceeded
    if (config_.max_cells > 0 &&
        cells_.size() > static_cast<size_t>(config_.max_cells)) {
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
            // Per-class decay: high-confidence classes decay slower.
            // This preserves well-established semantic labels while
            // allowing uncertain cells to return towards the prior.
            VectorXd probs = cell.state.getProbabilities();
            for (int c = 0; c < cell.state.log_odds.size(); ++c) {
                // Effective decay is reduced proportionally to class probability:
                //   decay_c = decay_rate * (1 - prob_c)
                // So the dominant class (prob ~1) barely decays, while low-
                // probability classes decay at nearly the full rate.
                double class_decay = decay_rate * (1.0 - probs(c));
                cell.state.log_odds(c) *= (1.0 - class_decay);
            }

            // Also decay dynamic status transitions
            cell.dynamic_status.decayTransitions(current_time);
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
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    std::unique_lock<std::shared_mutex> lock(mutex_);
    cells_.clear();
    total_observations_ = 0;

    std::string line;
    bool in_cells = false;
    Vector3d pos = Vector3d::Zero();
    int cls = 0;
    double conf = 0.0;
    int obs = 0;
    bool have_pos = false;

    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line[0] == '#') continue;

        // Header fields
        if (!in_cells) {
            if (line.rfind("resolution:", 0) == 0)
                config_.resolution = std::stod(line.substr(12));
            else if (line.rfind("num_classes:", 0) == 0)
                config_.num_classes = std::stoi(line.substr(13));
            else if (line.rfind("cells:", 0) == 0)
                in_cells = true;
            continue;
        }

        // Inside cells section
        if (line.rfind("- pos: [", 0) == 0) {
            // Flush previous cell if we have one
            if (have_pos) {
                MapCell& cell = getOrCreateCell(pos);
                // Set predicted class via log-odds bump
                if (cls >= 0 && cls < config_.num_classes) {
                    cell.state.log_odds(cls) += 5.0;  // Strong prior
                    cell.state.log_odds(cls) = clampLogOdds(cell.state.log_odds(cls));
                }
                cell.state.observation_count = obs;
                cell.updateFunctionalAttributes(0.0);
            }
            // Parse "- pos: [x, y, z]"
            size_t bracket = line.find('[');
            size_t end_bracket = line.find(']');
            if (bracket != std::string::npos && end_bracket != std::string::npos) {
                std::string coords = line.substr(bracket + 1, end_bracket - bracket - 1);
                std::istringstream ss(coords);
                char comma;
                ss >> pos.x() >> comma >> pos.y() >> comma >> pos.z();
                have_pos = true;
            }
        } else if (line.rfind("class:", 0) == 0) {
            cls = std::stoi(line.substr(7));
        } else if (line.rfind("confidence:", 0) == 0) {
            conf = std::stod(line.substr(12));
            (void)conf;  // confidence is derived from log-odds
        } else if (line.rfind("observations:", 0) == 0) {
            obs = std::stoi(line.substr(14));
        }
    }

    // Flush last cell
    if (have_pos) {
        MapCell& cell = getOrCreateCell(pos);
        if (cls >= 0 && cls < config_.num_classes) {
            cell.state.log_odds(cls) += 5.0;
            cell.state.log_odds(cls) = clampLogOdds(cell.state.log_odds(cls));
        }
        cell.state.observation_count = obs;
        cell.updateFunctionalAttributes(0.0);
    }

    total_observations_ = cells_.size();
    file.close();
    return true;
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
