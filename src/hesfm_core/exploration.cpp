/**
 * @file exploration.cpp
 * @brief Implementation of EMI-based exploration
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include "hesfm/exploration.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>

namespace hesfm {

// =============================================================================
// Constructors
// =============================================================================

ExplorationPlanner::ExplorationPlanner()
    : config_() {}

ExplorationPlanner::ExplorationPlanner(const ExplorationConfig& config)
    : config_(config) {}

// =============================================================================
// Main Interface
// =============================================================================

std::vector<ExplorationGoal> ExplorationPlanner::computeGoals(
    const SemanticMap& map,
    const Vector3d& robot_position,
    const Quaterniond& robot_orientation) {
    
    // Detect frontiers
    auto frontiers = detectFrontiers(map);
    
    if (frontiers.empty()) {
        return {};
    }
    
    // Sample candidate viewpoints
    auto candidates = sampleViewpoints(frontiers, robot_position);
    
    // Evaluate each candidate
    std::vector<ExplorationGoal> goals;
    goals.reserve(candidates.size());
    
    for (const auto& [position, orientation] : candidates) {
        ExplorationGoal goal;
        goal.position = position;
        goal.orientation = orientation;
        
        // Compute EMI
        goal.emi_value = computeEMI(map, position, orientation);
        goal.expected_info_gain = goal.emi_value;
        
        // Compute uncertainty reduction
        goal.uncertainty_reduction = computeUncertaintyReduction(map, position, orientation);
        
        // Compute distance
        goal.distance = (position - robot_position).norm();
        
        // Check reachability
        goal.is_reachable = isGoalReachable(map, position, robot_position);
        
        // Check safety
        goal.is_valid = isGoalSafe(map, position);
        
        // Compute utility
        goal.utility_score = computeUtility(goal, robot_position);
        
        if (goal.is_reachable && goal.is_valid) {
            goals.push_back(goal);
        }
    }
    
    // Sort by utility (descending)
    std::sort(goals.begin(), goals.end());
    
    // Assign ranks
    for (size_t i = 0; i < goals.size(); ++i) {
        goals[i].rank = static_cast<int>(i + 1);
    }
    
    // Limit number of goals
    if (goals.size() > static_cast<size_t>(config_.max_goals)) {
        goals.resize(config_.max_goals);
    }
    
    return goals;
}

std::optional<ExplorationGoal> ExplorationPlanner::getBestGoal(
    const SemanticMap& map,
    const Vector3d& robot_position,
    const Quaterniond& robot_orientation) {
    
    auto goals = computeGoals(map, robot_position, robot_orientation);
    
    if (goals.empty()) {
        return std::nullopt;
    }
    
    return goals[0];
}

bool ExplorationPlanner::isExplorationComplete(const SemanticMap& map) const {
    double coverage = map.getCoverage();
    double mean_uncertainty = map.getMeanUncertainty();
    
    // Exploration complete when coverage is high and uncertainty is low
    return coverage > 0.8 && mean_uncertainty < 0.2;
}

// =============================================================================
// Frontier Detection
// =============================================================================

std::vector<Frontier> ExplorationPlanner::detectFrontiers(const SemanticMap& map) const {
    auto cells = getFrontierCells(map);
    return clusterFrontiers(cells);
}

std::vector<FrontierCell> ExplorationPlanner::getFrontierCells(const SemanticMap& map) const {
    std::vector<FrontierCell> frontier_cells;
    
    auto all_cells = map.getOccupiedCells();
    const auto& config = map.getConfig();
    
    // For each observed cell, check if it's adjacent to unknown space
    for (const auto& cell : all_cells) {
        // Check 6 neighbors (face-connected)
        std::array<Vector3d, 6> neighbors = {{
            {cell.position.x() + config.resolution, cell.position.y(), cell.position.z()},
            {cell.position.x() - config.resolution, cell.position.y(), cell.position.z()},
            {cell.position.x(), cell.position.y() + config.resolution, cell.position.z()},
            {cell.position.x(), cell.position.y() - config.resolution, cell.position.z()},
            {cell.position.x(), cell.position.y(), cell.position.z() + config.resolution},
            {cell.position.x(), cell.position.y(), cell.position.z() - config.resolution}
        }};
        
        int unknown_neighbors = 0;
        for (const auto& neighbor_pos : neighbors) {
            auto neighbor = map.getCell(neighbor_pos);
            if (!neighbor || neighbor->state.observation_count == 0) {
                unknown_neighbors++;
            }
        }
        
        // Cell is a frontier if it has unknown neighbors
        if (unknown_neighbors > 0 && cell.isTraversable()) {
            FrontierCell fc;
            fc.position = cell.position;
            fc.frontier_id = -1;  // Will be assigned during clustering
            fc.information_gain = static_cast<double>(unknown_neighbors);
            fc.num_unknown_neighbors = unknown_neighbors;
            frontier_cells.push_back(fc);
        }
    }
    
    return frontier_cells;
}

std::vector<Frontier> ExplorationPlanner::clusterFrontiers(
    const std::vector<FrontierCell>& cells) const {
    
    if (cells.empty()) {
        return {};
    }
    
    std::vector<Frontier> frontiers;
    std::vector<bool> visited(cells.size(), false);
    int frontier_id = 0;
    
    // Simple connectivity-based clustering
    for (size_t i = 0; i < cells.size(); ++i) {
        if (visited[i]) continue;
        
        Frontier frontier;
        frontier.id = frontier_id;
        frontier.total_info_gain = 0.0;
        
        // BFS to find connected cells
        std::queue<size_t> queue;
        queue.push(i);
        visited[i] = true;
        
        while (!queue.empty()) {
            size_t curr = queue.front();
            queue.pop();
            
            FrontierCell fc = cells[curr];
            fc.frontier_id = frontier_id;
            frontier.cells.push_back(fc);
            frontier.total_info_gain += fc.information_gain;
            
            // Find neighbors
            for (size_t j = 0; j < cells.size(); ++j) {
                if (visited[j]) continue;
                
                double dist = (cells[j].position - cells[curr].position).norm();
                if (dist < 0.15) {  // Adjacent cells
                    visited[j] = true;
                    queue.push(j);
                }
            }
        }
        
        // Compute frontier statistics
        if (!frontier.cells.empty()) {
            frontier.size = static_cast<double>(frontier.cells.size());
            frontier.mean_info_gain = frontier.total_info_gain / frontier.size;
            
            // Compute centroid
            Vector3d sum = Vector3d::Zero();
            for (const auto& fc : frontier.cells) {
                sum += fc.position;
            }
            frontier.centroid = sum / frontier.size;
            
            // Only keep significant frontiers
            if (frontier.size >= config_.min_frontier_size) {
                frontiers.push_back(frontier);
            }
        }
        
        frontier_id++;
    }
    
    return frontiers;
}

// =============================================================================
// Information Gain Computation
// =============================================================================

double ExplorationPlanner::computeEMI(const SemanticMap& map,
                                        const Vector3d& viewpoint,
                                        const Quaterniond& orientation) const {
    
    double emi = 0.0;
    
    // Get visible cells
    auto visible_cells = getVisibleCells(map, viewpoint, orientation);
    
    for (const auto& cell_pos : visible_cells) {
        auto cell = map.getCell(cell_pos);
        
        // Unknown cells contribute geometric entropy
        if (!cell || cell->state.observation_count == 0) {
            emi += 1.0;  // Maximum entropy for unknown
            continue;
        }
        
        // Known cells contribute semantic entropy
        double semantic_entropy = cell->state.getNormalizedEntropy();
        emi += semantic_entropy;
    }
    
    return emi;
}

double ExplorationPlanner::computeGeometricInfoGain(
    const SemanticMap& map,
    const Vector3d& viewpoint,
    const Quaterniond& orientation) const {
    
    auto visible_cells = getVisibleCells(map, viewpoint, orientation);
    
    int unknown_count = 0;
    for (const auto& cell_pos : visible_cells) {
        auto cell = map.getCell(cell_pos);
        if (!cell || cell->state.observation_count == 0) {
            unknown_count++;
        }
    }
    
    return static_cast<double>(unknown_count);
}

double ExplorationPlanner::computeSemanticInfoGain(
    const SemanticMap& map,
    const Vector3d& viewpoint,
    const Quaterniond& orientation) const {
    
    auto visible_cells = getVisibleCells(map, viewpoint, orientation);
    
    double total_entropy = 0.0;
    int count = 0;
    
    for (const auto& cell_pos : visible_cells) {
        auto cell = map.getCell(cell_pos);
        if (cell && cell->state.observation_count > 0) {
            total_entropy += cell->state.getNormalizedEntropy();
            count++;
        }
    }
    
    return (count > 0) ? (total_entropy / count) : 0.0;
}

double ExplorationPlanner::computeUncertaintyReduction(
    const SemanticMap& map,
    const Vector3d& viewpoint,
    const Quaterniond& orientation) const {
    
    auto visible_cells = getVisibleCells(map, viewpoint, orientation);
    
    double current_uncertainty = 0.0;
    int count = 0;
    
    for (const auto& cell_pos : visible_cells) {
        auto cell = map.getCell(cell_pos);
        if (cell) {
            current_uncertainty += 1.0 - cell->state.getConfidence();
            count++;
        } else {
            current_uncertainty += 1.0;  // Unknown = maximum uncertainty
            count++;
        }
    }
    
    // Potential reduction assuming observation reduces uncertainty by ~50%
    return (count > 0) ? (0.5 * current_uncertainty / count) : 0.0;
}

// =============================================================================
// Visibility Computation
// =============================================================================

std::vector<Vector3d> ExplorationPlanner::getVisibleCells(
    const SemanticMap& map,
    const Vector3d& viewpoint,
    const Quaterniond& orientation) const {
    
    std::vector<Vector3d> visible;
    const auto& config = map.getConfig();
    
    // Sample cells within sensor range
    for (double x = viewpoint.x() - config_.sensor_range;
         x <= viewpoint.x() + config_.sensor_range;
         x += config.resolution) {
        for (double y = viewpoint.y() - config_.sensor_range;
             y <= viewpoint.y() + config_.sensor_range;
             y += config.resolution) {
            for (double z = viewpoint.z() - 1.0;  // Limited vertical range
                 z <= viewpoint.z() + 1.0;
                 z += config.resolution) {
                
                Vector3d cell_pos(x, y, z);
                
                // Check if in FOV
                if (!isInFOV(cell_pos, viewpoint, orientation)) {
                    continue;
                }
                
                // Check if visible (no occlusion)
                if (isCellVisible(map, cell_pos, viewpoint, orientation)) {
                    visible.push_back(cell_pos);
                }
            }
        }
    }
    
    return visible;
}

bool ExplorationPlanner::isCellVisible(
    const SemanticMap& map,
    const Vector3d& cell_position,
    const Vector3d& viewpoint,
    const Quaterniond& orientation) const {
    
    // Check if in FOV
    if (!isInFOV(cell_position, viewpoint, orientation)) {
        return false;
    }
    
    // Check distance
    double distance = (cell_position - viewpoint).norm();
    if (distance > config_.sensor_range) {
        return false;
    }
    
    // Raycast for occlusion
    return raycast(map, viewpoint, cell_position);
}

bool ExplorationPlanner::raycast(const SemanticMap& map,
                                   const Vector3d& start,
                                   const Vector3d& end) const {
    
    Vector3d direction = end - start;
    double length = direction.norm();
    
    if (length < EPSILON) {
        return true;
    }
    
    direction /= length;
    double step = map.getConfig().resolution * 0.5;
    
    for (double t = step; t < length - step; t += step) {
        Vector3d pos = start + direction * t;
        
        if (map.isObstacle(pos)) {
            return false;  // Occluded
        }
    }
    
    return true;  // Visible
}

// =============================================================================
// Goal Evaluation
// =============================================================================

double ExplorationPlanner::computeUtility(const ExplorationGoal& goal,
                                            const Vector3d& robot_position) const {
    
    // Weighted combination of factors
    double utility = 0.0;
    
    // Information gain (higher is better)
    utility += config_.weight_info_gain * goal.emi_value;
    
    // Distance penalty (closer is better)
    double normalized_distance = goal.distance / config_.max_distance;
    utility -= config_.weight_distance * normalized_distance;
    
    // Uncertainty reduction (higher is better)
    utility += config_.weight_uncertainty * goal.uncertainty_reduction;
    
    return utility;
}

bool ExplorationPlanner::isGoalReachable(const SemanticMap& map,
                                           const Vector3d& goal,
                                           const Vector3d& robot_position) const {
    
    // Simple check: path is clear (no obstacles)
    // A proper implementation would use a path planner
    return raycast(map, robot_position, goal);
}

bool ExplorationPlanner::isGoalSafe(const SemanticMap& map, const Vector3d& goal) const {
    // Check distance to nearest obstacle
    const auto& config = map.getConfig();
    
    // Sample nearby cells
    for (double dx = -config_.min_obstacle_distance;
         dx <= config_.min_obstacle_distance;
         dx += config.resolution) {
        for (double dy = -config_.min_obstacle_distance;
             dy <= config_.min_obstacle_distance;
             dy += config.resolution) {
            
            Vector3d nearby(goal.x() + dx, goal.y() + dy, goal.z());
            
            if (map.isObstacle(nearby)) {
                return false;
            }
        }
    }
    
    return true;
}

// =============================================================================
// Private Methods
// =============================================================================

std::vector<std::pair<Vector3d, Quaterniond>> ExplorationPlanner::sampleViewpoints(
    const std::vector<Frontier>& frontiers,
    const Vector3d& robot_position) const {
    
    std::vector<std::pair<Vector3d, Quaterniond>> viewpoints;
    
    for (const auto& frontier : frontiers) {
        // Sample viewpoints around frontier centroid
        double distance = config_.sensor_range * 0.7;  // View from 70% of max range
        
        // Sample at different angles
        for (double angle = 0; angle < 2 * M_PI; angle += M_PI / 4) {
            Vector3d offset(distance * std::cos(angle),
                           distance * std::sin(angle),
                           0.0);
            
            Vector3d viewpoint = frontier.centroid - offset;
            
            // Keep same height as robot
            viewpoint.z() = robot_position.z();
            
            // Compute viewing direction towards frontier
            Quaterniond orientation = computeViewingDirection(viewpoint, frontier.centroid);
            
            viewpoints.emplace_back(viewpoint, orientation);
        }
    }
    
    return viewpoints;
}

Quaterniond ExplorationPlanner::computeViewingDirection(
    const Vector3d& from,
    const Vector3d& target) const {
    
    Vector3d direction = (target - from).normalized();
    
    // Compute rotation from forward (1,0,0) to direction
    Vector3d forward(1, 0, 0);
    
    Vector3d axis = forward.cross(direction);
    double angle = std::acos(std::clamp(forward.dot(direction), -1.0, 1.0));
    
    if (axis.norm() < EPSILON) {
        return Quaterniond::Identity();
    }
    
    axis.normalize();
    return Quaterniond(Eigen::AngleAxisd(angle, axis));
}

bool ExplorationPlanner::isInFOV(const Vector3d& point,
                                   const Vector3d& viewpoint,
                                   const Quaterniond& orientation) const {
    
    Vector3d direction = point - viewpoint;
    double distance = direction.norm();
    
    if (distance < EPSILON || distance > config_.sensor_range) {
        return false;
    }
    
    direction.normalize();
    
    // Get camera forward direction
    Vector3d forward = orientation * Vector3d(1, 0, 0);
    
    // Check horizontal angle
    Vector3d horizontal_dir(direction.x(), direction.y(), 0);
    horizontal_dir.normalize();
    Vector3d horizontal_fwd(forward.x(), forward.y(), 0);
    horizontal_fwd.normalize();
    
    double cos_h = horizontal_dir.dot(horizontal_fwd);
    double angle_h = std::acos(std::clamp(cos_h, -1.0, 1.0));
    
    if (angle_h > config_.sensor_fov_horizontal / 2) {
        return false;
    }
    
    // Check vertical angle
    double angle_v = std::asin(std::clamp(direction.z(), -1.0, 1.0));
    
    if (std::abs(angle_v) > config_.sensor_fov_vertical / 2) {
        return false;
    }
    
    return true;
}

} // namespace hesfm
