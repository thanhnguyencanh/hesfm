/**
 * @file types.h
 * @brief Core type definitions for HESFM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This file contains all fundamental data structures used throughout
 * the HESFM framework, including semantic points, Gaussian primitives,
 * semantic states, and configuration structures.
 */

#ifndef HESFM_TYPES_H_
#define HESFM_TYPES_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <cmath>

namespace hesfm {

// =============================================================================
// Eigen Type Aliases
// =============================================================================

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using VectorXd = Eigen::VectorXd;
using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using MatrixXd = Eigen::MatrixXd;
using Quaterniond = Eigen::Quaterniond;

// =============================================================================
// Constants
// =============================================================================

/// Default number of semantic classes (NYUv2 40-class)
constexpr int DEFAULT_NUM_CLASSES = 37;

/// Number of semantic classes for each dataset
constexpr int NYUV2_NUM_CLASSES   = 40;
constexpr int SUNRGBD_NUM_CLASSES = 37;

/// Maximum log-odds value to prevent overflow
constexpr double LOG_ODDS_MAX = 10.0;

/// Minimum log-odds value to prevent underflow
constexpr double LOG_ODDS_MIN = -10.0;

/// Small epsilon for numerical stability
constexpr double EPSILON = 1e-10;

/// NYUv2 class names (40 classes, indices 0-39, void excluded)
const std::vector<std::string> NYUV2_CLASS_NAMES = {
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "blinds",
    "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor_mat", "clothes", "ceiling", "books", "fridge",
    "television", "paper", "towel", "shower_curtain", "box",
    "whiteboard", "person", "night_stand", "toilet", "sink",
    "lamp", "bathtub", "bag", "otherstructure", "otherfurniture",
    "otherprop"
};

/// SUN RGB-D class names (37 classes, indices 0-36, void excluded)
/// Order matches ESANet CLASS_NAMES_ENGLISH[1:] (index 0 = void skipped)
const std::vector<std::string> SUNRGBD_CLASS_NAMES = {
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "blinds",
    "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor_mat", "clothes", "ceiling", "books", "fridge",
    "tv", "paper", "towel", "shower_curtain", "box",
    "whiteboard", "person", "night_stand", "toilet", "sink",
    "lamp", "bathtub", "bag"
};

/// NYUv2 class colors (RGB) — must match NYUv2ColorPalette.COLORS in semantic_segmentation_node.py
const std::vector<std::array<uint8_t,3>> NYUV2_CLASS_COLORS = {
    {128,128,128},{139,119,101},{244,164, 96},{255,182,193},{255,215,  0},
    {220, 20, 60},{255,140,  0},{139, 69, 19},{135,206,235},{160, 82, 45},
    {255,105,180},{  0,128,128},{210,180,140},{ 70,130,180},{188,143,143},
    {147,112,219},{222,184,135},{255,228,225},{192,192,192},{139,119,101},
    {128,  0,128},{245,245,245},{139, 90, 43},{173,216,230},{  0,  0,139},
    {255,255,224},{240,255,255},{176,224,230},{210,105, 30},{255,255,255},
    {255,  0,  0},{ 85,107, 47},{255,255,240},{176,196,222},{255,250,205},
    {224,255,255},{ 75,  0,130},{169,169,169},{105,105,105},{128,128,  0}
};

/// SUN RGB-D class colors (RGB) — must match SUNRGBDColorPalette.COLORS in semantic_segmentation_node.py
const std::vector<std::array<uint8_t,3>> SUNRGBD_CLASS_COLORS = {
    {119,119,119},{244,243,131},{137, 28,157},{150,255,255},{ 54,114,113},
    {  0,  0,176},{255, 69,  0},{ 87,112,255},{  0,163, 33},{255,150,255},
    {255,180, 10},{101, 70, 86},{ 38,230,  0},{255,120, 70},{117, 41,121},
    {150,255,  0},{132,  0,255},{ 24,209,255},{191,130, 35},{219,200,109},
    {154, 62, 86},{255,190,190},{255,  0,255},{152,163, 55},{192, 79,212},
    {230,230,230},{ 53,130, 64},{155,249,152},{ 87, 64, 34},{214,209,175},
    {170,  0, 59},{255,  0,  0},{193,195,234},{ 70, 72,115},{255,255,  0},
    { 52, 57,131},{ 12, 83, 45}
};

/// NYUv2 traversable class indices (floor=1, floor_mat=19)
const std::set<int> DEFAULT_TRAVERSABLE_CLASSES = {1, 19};

/// SUN RGB-D traversable class indices (floor=1, floor_mat=19)
const std::set<int> SUNRGBD_TRAVERSABLE_CLASSES = {1, 19};

// =============================================================================
// Affordance Types
// =============================================================================

/**
 * @brief Affordance categories for semantic classes
 *
 * Each semantic class maps to a set of affordances describing what
 * actions the robot can perform at/with that region.
 */
enum class AffordanceType : uint8_t {
    TRAVERSABLE   = 0,  ///< Robot can drive over this surface
    PLACEABLE     = 1,  ///< Robot can place objects on this surface
    OPENABLE      = 2,  ///< Robot can open this (door, drawer, fridge)
    SITTABLE      = 3,  ///< Object supports sitting (chair, sofa, bed)
    SUPPORTIVE    = 4,  ///< Structural support (wall, ceiling, column)
    CONTAINABLE   = 5,  ///< Can contain/store objects (cabinet, shelf, box)
    MANIPULABLE   = 6,  ///< Robot can pick up / move (bag, box, books, clothes)
    VIEWABLE      = 7,  ///< Information source (tv, whiteboard, picture, mirror)
    AVOIDABLE     = 8,  ///< Dynamic/dangerous, must avoid (person)
    NUM_AFFORDANCES = 9
};

/// Number of affordance types
constexpr int NUM_AFFORDANCE_TYPES = static_cast<int>(AffordanceType::NUM_AFFORDANCES);

/**
 * @brief Static lookup table: semantic class index -> bitset of affordances
 *
 * SUN RGB-D 37-class mapping. Each entry is a vector of AffordanceType values
 * that apply to that class. The index matches SUNRGBD_CLASS_NAMES.
 */
inline std::vector<std::vector<AffordanceType>> getDefaultAffordanceTable() {
    std::vector<std::vector<AffordanceType>> table(SUNRGBD_NUM_CLASSES);
    // 0: wall           -> SUPPORTIVE
    table[0]  = {AffordanceType::SUPPORTIVE};
    // 1: floor          -> TRAVERSABLE
    table[1]  = {AffordanceType::TRAVERSABLE};
    // 2: cabinet        -> CONTAINABLE, OPENABLE
    table[2]  = {AffordanceType::CONTAINABLE, AffordanceType::OPENABLE};
    // 3: bed            -> SITTABLE, PLACEABLE
    table[3]  = {AffordanceType::SITTABLE, AffordanceType::PLACEABLE};
    // 4: chair          -> SITTABLE
    table[4]  = {AffordanceType::SITTABLE};
    // 5: sofa           -> SITTABLE
    table[5]  = {AffordanceType::SITTABLE};
    // 6: table          -> PLACEABLE
    table[6]  = {AffordanceType::PLACEABLE};
    // 7: door           -> OPENABLE
    table[7]  = {AffordanceType::OPENABLE};
    // 8: window         -> SUPPORTIVE
    table[8]  = {AffordanceType::SUPPORTIVE};
    // 9: bookshelf      -> CONTAINABLE
    table[9]  = {AffordanceType::CONTAINABLE};
    // 10: picture       -> VIEWABLE
    table[10] = {AffordanceType::VIEWABLE};
    // 11: counter       -> PLACEABLE
    table[11] = {AffordanceType::PLACEABLE};
    // 12: blinds        -> (none)
    table[12] = {};
    // 13: desk          -> PLACEABLE
    table[13] = {AffordanceType::PLACEABLE};
    // 14: shelves       -> CONTAINABLE
    table[14] = {AffordanceType::CONTAINABLE};
    // 15: curtain       -> (none)
    table[15] = {};
    // 16: dresser       -> CONTAINABLE, OPENABLE, PLACEABLE
    table[16] = {AffordanceType::CONTAINABLE, AffordanceType::OPENABLE, AffordanceType::PLACEABLE};
    // 17: pillow        -> MANIPULABLE
    table[17] = {AffordanceType::MANIPULABLE};
    // 18: mirror        -> VIEWABLE
    table[18] = {AffordanceType::VIEWABLE};
    // 19: floor_mat     -> TRAVERSABLE
    table[19] = {AffordanceType::TRAVERSABLE};
    // 20: clothes       -> MANIPULABLE
    table[20] = {AffordanceType::MANIPULABLE};
    // 21: ceiling       -> SUPPORTIVE
    table[21] = {AffordanceType::SUPPORTIVE};
    // 22: books         -> MANIPULABLE
    table[22] = {AffordanceType::MANIPULABLE};
    // 23: fridge        -> CONTAINABLE, OPENABLE
    table[23] = {AffordanceType::CONTAINABLE, AffordanceType::OPENABLE};
    // 24: tv            -> VIEWABLE
    table[24] = {AffordanceType::VIEWABLE};
    // 25: paper         -> MANIPULABLE
    table[25] = {AffordanceType::MANIPULABLE};
    // 26: towel         -> MANIPULABLE
    table[26] = {AffordanceType::MANIPULABLE};
    // 27: shower_curtain -> (none)
    table[27] = {};
    // 28: box           -> CONTAINABLE, MANIPULABLE
    table[28] = {AffordanceType::CONTAINABLE, AffordanceType::MANIPULABLE};
    // 29: whiteboard    -> VIEWABLE
    table[29] = {AffordanceType::VIEWABLE};
    // 30: person        -> AVOIDABLE
    table[30] = {AffordanceType::AVOIDABLE};
    // 31: night_stand   -> PLACEABLE, CONTAINABLE
    table[31] = {AffordanceType::PLACEABLE, AffordanceType::CONTAINABLE};
    // 32: toilet        -> (none)
    table[32] = {};
    // 33: sink          -> (none)
    table[33] = {};
    // 34: lamp          -> (none)
    table[34] = {};
    // 35: bathtub       -> CONTAINABLE
    table[35] = {AffordanceType::CONTAINABLE};
    // 36: bag           -> CONTAINABLE, MANIPULABLE
    table[36] = {AffordanceType::CONTAINABLE, AffordanceType::MANIPULABLE};
    return table;
}

/// Human-readable affordance names
inline const char* affordanceTypeName(AffordanceType t) {
    switch (t) {
        case AffordanceType::TRAVERSABLE:  return "traversable";
        case AffordanceType::PLACEABLE:    return "placeable";
        case AffordanceType::OPENABLE:     return "openable";
        case AffordanceType::SITTABLE:     return "sittable";
        case AffordanceType::SUPPORTIVE:   return "supportive";
        case AffordanceType::CONTAINABLE:  return "containable";
        case AffordanceType::MANIPULABLE:  return "manipulable";
        case AffordanceType::VIEWABLE:     return "viewable";
        case AffordanceType::AVOIDABLE:    return "avoidable";
        default:                           return "unknown";
    }
}

// =============================================================================
// Dynamic Object Status
// =============================================================================

/**
 * @brief Tracks dynamic status of a map cell over time
 *
 * Uses a sliding-window approach: stores the last N predicted classes
 * and timestamps to detect changes (dynamic objects) versus stable regions.
 */
struct DynamicObjectStatus {
    /// Has the cell's dominant class changed recently?
    bool is_dynamic = false;

    /// Number of class transitions detected in the observation window
    int transition_count = 0;

    /// Previous dominant class (for change detection)
    int previous_class = -1;

    /// Timestamp when the class last changed
    double last_transition_time = 0.0;

    /// Occupancy confidence: exponential moving average of occupancy observations
    /// 1.0 = always occupied, 0.0 = never occupied
    double occupancy_confidence = 0.0;

    /// Number of frames this cell was observed
    int observation_frames = 0;

    /// Number of frames this cell was occupied
    int occupied_frames = 0;

    // -------------------------------------------------------------------------
    // Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Update dynamic status with a new observation
     * @param current_class   Predicted class from this frame
     * @param current_time    Current timestamp
     * @param transition_window  Time window for counting transitions (seconds)
     * @param ema_alpha       Exponential moving average smoothing factor
     */
    void update(int current_class, double current_time,
                double transition_window = 10.0, double ema_alpha = 0.1) {

        observation_frames++;

        // Detect class transition
        if (previous_class >= 0 && current_class != previous_class) {
            // Only count as transition if within the window
            if (current_time - last_transition_time < transition_window) {
                transition_count++;
            }
            last_transition_time = current_time;
        }

        // Mark dynamic if enough transitions observed
        // Threshold: 2+ transitions within the window
        is_dynamic = (transition_count >= 2);

        // Update occupancy confidence via EMA
        double occupied = (current_class >= 0) ? 1.0 : 0.0;
        occupancy_confidence = ema_alpha * occupied
                             + (1.0 - ema_alpha) * occupancy_confidence;

        previous_class = current_class;
    }

    /**
     * @brief Decay transition count over time (call periodically)
     */
    void decayTransitions(double current_time, double transition_window = 10.0) {
        if (current_time - last_transition_time > transition_window) {
            transition_count = std::max(0, transition_count - 1);
            if (transition_count < 2) {
                is_dynamic = false;
            }
        }
    }

    /**
     * @brief Reset dynamic status
     */
    void reset() {
        is_dynamic = false;
        transition_count = 0;
        previous_class = -1;
        last_transition_time = 0.0;
        occupancy_confidence = 0.0;
        observation_frames = 0;
        occupied_frames = 0;
    }
};

// =============================================================================
// Reachability Information
// =============================================================================

/**
 * @brief Reachability state for a map cell
 *
 * Tracks whether a cell is reachable by the robot, considering traversability
 * of surrounding cells and geometric constraints. Reachability influences the
 * adaptive kernel gating: unreachable areas receive lower update weight.
 */
struct ReachabilityInfo {
    /// Is this cell reachable from the robot's current position?
    bool is_reachable = false;

    /// Reachability probability [0, 1] — 1.0 = certainly reachable
    double probability = 0.5;

    /// Shortest traversable distance from robot (meters); INF if unreachable
    double distance = std::numeric_limits<double>::infinity();

    /// Timestamp of last reachability evaluation
    double last_evaluated = 0.0;

    /// Number of times this cell was evaluated as reachable
    int reachable_count = 0;

    /// Total number of reachability evaluations
    int evaluation_count = 0;

    void update(bool reachable, double dist, double current_time) {
        evaluation_count++;
        if (reachable) {
            reachable_count++;
            distance = std::min(distance, dist);
        }
        probability = static_cast<double>(reachable_count) / evaluation_count;
        is_reachable = (probability > 0.5);
        last_evaluated = current_time;
    }

    void reset() {
        is_reachable = false;
        probability = 0.5;
        distance = std::numeric_limits<double>::infinity();
        last_evaluated = 0.0;
        reachable_count = 0;
        evaluation_count = 0;
    }
};

// =============================================================================
// Affordance Bitset
// =============================================================================

/**
 * @brief Compact bitset encoding active affordances for a map cell.
 *
 * Packed into a uint16_t for memory efficiency (one bit per AffordanceType).
 */
struct AffordanceBitset {
    uint16_t bits = 0;

    bool has(AffordanceType t) const { return (bits >> static_cast<uint8_t>(t)) & 1; }
    void set(AffordanceType t)       { bits |= (1 << static_cast<uint8_t>(t)); }
    void clear(AffordanceType t)     { bits &= ~(1 << static_cast<uint8_t>(t)); }
    void clearAll()                  { bits = 0; }

    /// Populate from semantic class using the default affordance table
    void fromSemanticClass(int cls) {
        clearAll();
        auto table = getDefaultAffordanceTable();
        if (cls >= 0 && cls < static_cast<int>(table.size())) {
            for (auto aff : table[cls]) {
                set(aff);
            }
        }
    }

    /// Count number of active affordances
    int count() const {
        int c = 0;
        for (int i = 0; i < NUM_AFFORDANCE_TYPES; ++i)
            c += (bits >> i) & 1;
        return c;
    }
};

// =============================================================================
// Uncertainty Decomposition Structure
// =============================================================================

/**
 * @brief Multi-source uncertainty decomposition result
 * 
 * Contains individual uncertainty components and the weighted total:
 * U_total = w_sem*U_sem + w_spa*U_spa + w_obs*U_obs + w_temp*U_temp
 */
struct UncertaintyDecomposition {
    double semantic = 0.0;      ///< Semantic uncertainty from EDL: U = K/S
    double spatial = 0.0;       ///< Spatial consistency uncertainty
    double observation = 0.0;   ///< Sensor model uncertainty
    double temporal = 0.0;      ///< Temporal prediction consistency
    double total = 0.0;         ///< Weighted combination
    
    UncertaintyDecomposition() = default;
    
    UncertaintyDecomposition(double sem, double spa, double obs, double temp, double tot)
        : semantic(sem), spatial(spa), observation(obs), temporal(temp), total(tot) {}
};

// =============================================================================
// Semantic Point Structure
// =============================================================================

/**
 * @brief A single 3D point with semantic label and uncertainty information
 * 
 * Represents point-level semantic information before aggregation into
 * Gaussian primitives. Contains full uncertainty decomposition.
 */
struct SemanticPoint {
    /// 3D position in world/sensor frame
    Vector3d position = Vector3d::Zero();
    
    /// Predicted semantic class (0 to num_classes-1)
    int semantic_class = 0;

    /// Confidence for predicted class (replaces full class_probabilities vector).
    /// Background probability = (1 - semantic_confidence) / (num_classes - 1).
    /// Full distribution is materialized on demand in GaussianPrimitiveBuilder.
    float semantic_confidence = 0.9f;
    
    /// Evidence values from EDL (optional, for computing semantic uncertainty)
    std::vector<double> evidence;
    
    /// Multi-source uncertainty components
    double uncertainty_semantic = 1.0;
    double uncertainty_spatial = 1.0;
    double uncertainty_observation = 1.0;
    double uncertainty_temporal = 0.5;
    double uncertainty_total = 1.0;
    
    /// Depth value from sensor (meters)
    double depth = 0.0;
    
    /// RGB color
    uint8_t r = 0, g = 0, b = 0;
    
    /// Surface normal (if available)
    Vector3d normal = Vector3d::Zero();
    
    /// Is this point on a traversable surface?
    bool is_traversable = false;
    
    /// Timestamp of observation
    double timestamp = 0.0;
    
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    SemanticPoint() = default;
    
    SemanticPoint(const Vector3d& pos, int cls, double unc = 1.0)
        : position(pos), semantic_class(cls), uncertainty_total(unc) {}
    
    // -------------------------------------------------------------------------
    // Utility Methods
    // -------------------------------------------------------------------------
    
    /**
     * @brief Get confidence as 1 - uncertainty
     */
    double getConfidence() const {
        return 1.0 - uncertainty_total;
    }
    
    /**
     * @brief Get weight for aggregation (inverse uncertainty)
     */
    double getWeight(double lambda = 1.0) const {
        return 1.0 / (uncertainty_total + lambda * 0.1);
    }
    
    /**
     * @brief Check if point has valid data
     */
    bool isValid() const {
        return std::isfinite(position.x()) && 
               std::isfinite(position.y()) && 
               std::isfinite(position.z()) &&
               depth > 0.0;
    }
};

// =============================================================================
// Gaussian Primitive Structure
// =============================================================================

/**
 * @brief A Gaussian primitive aggregating multiple semantic points
 * 
 * Core representation in the hierarchical HESFM framework:
 * G = (μ, Σ, p, u) where μ=centroid, Σ=covariance, p=class probs, u=uncertainty
 */
struct GaussianPrimitive {
    /// Unique identifier
    uint32_t id = 0;
    
    /// Centroid position (mean of aggregated points)
    Vector3d centroid = Vector3d::Zero();
    
    /// Covariance matrix (3x3)
    Matrix3d covariance = Matrix3d::Identity() * 0.01;
    
    /// Predicted semantic class
    int semantic_class = 0;
    
    /// Aggregated class probability distribution (DST fusion result)
    std::vector<double> class_probabilities;
    
    /// Aggregated uncertainty
    double uncertainty = 1.0;
    
    /// DST conflict measure: K = Σ_{A∩B=∅} m1(A)*m2(B)
    double conflict = 0.0;
    
    /// Number of points aggregated
    int point_count = 0;
    
    /// Sum of point weights
    double total_weight = 0.0;
    
    /// Adaptive kernel length scale
    double length_scale = 0.1;
    
    /// Is this primitive from a dynamic object?
    bool is_dynamic = false;
    
    /// Reachability probability [0, 1]
    double reachability = 0.5;
    
    /// Affordance values (application-specific)
    std::vector<double> affordances;
    
    /// Timestamp of creation
    double timestamp = 0.0;
    
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    GaussianPrimitive() = default;
    
    GaussianPrimitive(uint32_t id_, const Vector3d& cent, int cls)
        : id(id_), centroid(cent), semantic_class(cls) {}
    
    // -------------------------------------------------------------------------
    // Geometric Methods
    // -------------------------------------------------------------------------
    
    /**
     * @brief Get eigenvalues of covariance matrix
     */
    Vector3d getEigenvalues() const {
        Eigen::SelfAdjointEigenSolver<Matrix3d> solver(covariance);
        return solver.eigenvalues();
    }
    
    /**
     * @brief Get eigenvectors as quaternion (orientation)
     */
    Quaterniond getOrientation() const {
        Eigen::SelfAdjointEigenSolver<Matrix3d> solver(covariance);
        Matrix3d R = solver.eigenvectors();
        // Ensure right-handed coordinate system
        if (R.determinant() < 0) {
            R.col(2) *= -1;
        }
        return Quaterniond(R);
    }
    
    /**
     * @brief Compute adaptive length scale based on covariance
     * l = l_min + (l_max - l_min) * (trace(Σ) / max_trace)^(1/3)
     */
    double computeAdaptiveLengthScale(double l_min, double l_max, double max_trace) const {
        double trace = covariance.trace();
        double ratio = std::pow(trace / std::max(max_trace, EPSILON), 1.0/3.0);
        return l_min + (l_max - l_min) * std::clamp(ratio, 0.0, 1.0);
    }
    
    /**
     * @brief Compute Mahalanobis distance to a point
     */
    double mahalanobisDistance(const Vector3d& point) const {
        Vector3d diff = point - centroid;
        Matrix3d cov_inv = (covariance + EPSILON * Matrix3d::Identity()).inverse();
        return std::sqrt(diff.transpose() * cov_inv * diff);
    }
    
    /**
     * @brief Get volume of the ellipsoid (4/3 * π * √det(Σ))
     */
    double getVolume() const {
        return (4.0 / 3.0) * M_PI * std::sqrt(std::max(covariance.determinant(), EPSILON));
    }
    
    // -------------------------------------------------------------------------
    // Semantic Methods
    // -------------------------------------------------------------------------
    
    /**
     * @brief Get confidence (max probability)
     */
    double getConfidence() const {
        if (class_probabilities.empty()) return 0.0;
        return *std::max_element(class_probabilities.begin(), class_probabilities.end());
    }
    
    /**
     * @brief Get entropy of class distribution
     */
    double getEntropy() const {
        double entropy = 0.0;
        for (double p : class_probabilities) {
            if (p > EPSILON) {
                entropy -= p * std::log(p);
            }
        }
        return entropy;
    }
    
    /**
     * @brief Check if conflict is below threshold
     */
    bool isConsistent(double threshold = 0.3) const {
        return conflict < threshold;
    }
};

// =============================================================================
// Semantic State Structure
// =============================================================================

/**
 * @brief Extended semantic state for map cells
 * 
 * Uses log-odds representation for efficient Bayesian updates:
 * h_c = log(P(c) / P_prior(c))
 */
struct SemanticState {
    /// Log-odds values for each class
    VectorXd log_odds;
    
    /// Number of observations integrated
    int observation_count = 0;
    
    /// Timestamp of last update
    double last_update_time = 0.0;
    
    /// Is this cell associated with a dynamic object?
    bool is_dynamic = false;
    
    /// Reachability probability [0, 1]
    double reachability = 0.5;
    
    /// Affordance values
    std::vector<double> affordances;
    
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    SemanticState(int num_classes = DEFAULT_NUM_CLASSES)
        : log_odds(VectorXd::Zero(num_classes)) {}
    
    // -------------------------------------------------------------------------
    // Methods
    // -------------------------------------------------------------------------
    
    /**
     * @brief Get class probabilities via softmax
     */
    VectorXd getProbabilities() const {
        VectorXd exp_odds = log_odds.array().exp();
        return exp_odds / exp_odds.sum();
    }
    
    /**
     * @brief Get predicted class (argmax)
     */
    int getPredictedClass() const {
        int max_idx = 0;
        log_odds.maxCoeff(&max_idx);
        return max_idx;
    }
    
    /**
     * @brief Get prediction confidence
     */
    double getConfidence() const {
        return getProbabilities().maxCoeff();
    }
    
    /**
     * @brief Get entropy of distribution
     */
    double getEntropy() const {
        VectorXd probs = getProbabilities();
        double entropy = 0.0;
        for (int i = 0; i < probs.size(); ++i) {
            if (probs(i) > EPSILON) {
                entropy -= probs(i) * std::log(probs(i));
            }
        }
        return entropy;
    }
    
    /**
     * @brief Get maximum possible entropy (for normalization)
     */
    double getMaxEntropy() const {
        return std::log(static_cast<double>(log_odds.size()));
    }
    
    /**
     * @brief Get normalized entropy [0, 1]
     */
    double getNormalizedEntropy() const {
        return getEntropy() / getMaxEntropy();
    }
    
    /**
     * @brief Check if cell has been observed
     */
    bool isObserved() const {
        return observation_count > 0;
    }
    
    /**
     * @brief Reset state
     */
    void reset() {
        log_odds.setZero();
        observation_count = 0;
        last_update_time = 0.0;
        is_dynamic = false;
        reachability = 0.5;
        affordances.clear();
    }
};

// =============================================================================
// Map Cell Structure
// =============================================================================

/**
 * @brief A single cell in the semantic map
 */
struct MapCell {
    /// Grid indices
    int ix = 0, iy = 0, iz = 0;

    /// Center position in world frame
    Vector3d position = Vector3d::Zero();

    /// Semantic state
    SemanticState state;

    /// Navigation cost [-1=unknown, 0=free, 100=lethal]
    int8_t nav_cost = -1;

    /// Extended functional attributes (HESFM innovation #4)
    DynamicObjectStatus dynamic_status;
    ReachabilityInfo reachability;
    AffordanceBitset affordances;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    MapCell(int num_classes = DEFAULT_NUM_CLASSES) : state(num_classes) {}

    MapCell(int x, int y, int z, int num_classes = DEFAULT_NUM_CLASSES)
        : ix(x), iy(y), iz(z), state(num_classes) {}

    // -------------------------------------------------------------------------
    // Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Check if cell is traversable
     */
    bool isTraversable(const std::set<int>& traversable_classes = DEFAULT_TRAVERSABLE_CLASSES) const {
        if (!state.isObserved()) return false;
        int pred_class = state.getPredictedClass();
        return traversable_classes.count(pred_class) > 0;
    }

    /**
     * @brief Check if cell is an obstacle
     */
    bool isObstacle(double confidence_threshold = 0.5) const {
        if (!state.isObserved()) return false;
        return state.getConfidence() >= confidence_threshold && nav_cost == 100;
    }

    /**
     * @brief Update extended functional attributes from current semantic state
     */
    void updateFunctionalAttributes(double current_time) {
        int pred_class = state.getPredictedClass();
        // Update dynamic status
        dynamic_status.update(pred_class, current_time);
        // Update affordance bitset from predicted class
        affordances.fromSemanticClass(pred_class);
    }

    /**
     * @brief Get hash key for map storage
     */
    size_t getHash() const {
        size_t hash = 0;
        hash ^= std::hash<int>()(ix) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>()(iy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>()(iz) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

// =============================================================================
// Sensor Model Structure
// =============================================================================

/**
 * @brief Sensor observation model parameters
 */
struct SensorModel {
    /// Maximum reliable range (meters)
    double max_range = 6.0;
    
    /// Minimum range (meters)
    double min_range = 0.1;
    
    /// Range measurement standard deviation
    double range_sigma = 0.02;
    
    /// Angular measurement standard deviation
    double angular_sigma = 0.01;
    
    /// Radius for local density computation
    double density_radius = 0.1;
    
    /// Maximum expected point density
    double max_density = 100.0;
    
    /// Coefficients for observation uncertainty
    double sigma_range = 0.5;
    double sigma_density = 0.3;
    double sigma_angle = 0.2;
    
    /// Camera intrinsics
    double fx = 386.0;
    double fy = 386.0;
    double cx = 320.0;
    double cy = 240.0;
    int width = 640;
    int height = 480;
};

// =============================================================================
// Exploration Goal Structure
// =============================================================================

/**
 * @brief Exploration goal based on Extended Mutual Information
 */
struct ExplorationGoal {
    /// Goal position
    Vector3d position = Vector3d::Zero();
    
    /// Goal orientation (viewing direction)
    Quaterniond orientation = Quaterniond::Identity();
    
    /// Extended Mutual Information value
    double emi_value = 0.0;
    
    /// Expected information gain
    double expected_info_gain = 0.0;
    
    /// Uncertainty reduction potential
    double uncertainty_reduction = 0.0;
    
    /// Distance from current robot position
    double distance = 0.0;
    
    /// Overall utility score
    double utility_score = 0.0;
    
    /// Rank among candidates (1 = best)
    int rank = 0;
    
    /// Is this goal reachable?
    bool is_reachable = true;
    
    /// Is this goal still valid?
    bool is_valid = true;
    
    /// Frontier ID (if applicable)
    int frontier_id = -1;
    
    /// Number of unknown cells visible
    int num_unknown_visible = 0;
    
    /**
     * @brief Compare by utility (for sorting)
     */
    bool operator<(const ExplorationGoal& other) const {
        return utility_score > other.utility_score; // Higher is better
    }
};

} // namespace hesfm

#endif // HESFM_TYPES_H_