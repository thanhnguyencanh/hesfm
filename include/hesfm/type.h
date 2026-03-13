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

/// Default traversable class indices (floor=1, floor_mat=19)
const std::set<int> DEFAULT_TRAVERSABLE_CLASSES = {1, 19};

} // namespace hesfm

#endif // HESFM_TYPES_H_