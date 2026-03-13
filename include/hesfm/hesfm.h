/**
 * @file hesfm.h
 * @brief Main header file for HESFM (Hierarchical Evidential Semantic-Functional Mapping)
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * @mainpage HESFM: Hierarchical Evidential Semantic-Functional Mapping
 * 
 * @section intro_sec Introduction
 * 
 * HESFM is a novel semantic mapping framework for uncertainty-aware robot
 * navigation. It features:
 * 
 * - **Multi-source uncertainty decomposition**: Separates uncertainty into
 *   semantic, spatial, observation, and temporal components
 * - **Hierarchical Gaussian primitives**: Efficient aggregation of semantic
 *   points using uncertainty-weighted clustering and DST fusion
 * - **Adaptive anisotropic kernel**: Geometry-aware BKI with uncertainty
 *   gating and reachability constraints
 * - **Extended semantic state**: Beyond class probabilities - includes
 *   dynamic status, reachability, and affordances
 * - **EMI-based exploration**: Information-theoretic goal selection
 */
#ifndef HESFM_HESFM_H_
#define HESFM_HESFM_H_

// Version information
#define HESFM_VERSION_MAJOR 1
#define HESFM_VERSION_MINOR 0
#define HESFM_VERSION_PATCH 0
#define HESFM_VERSION_STRING "1.0.0"

// Core headers 
#include "hesfm/types.h"
#include "hesfm/config.h"

// Processing modules
#include "hesfm/uncertainty.h"
#include "hesfm/gaussian_primitive.h"
#include "hesfm/adaptive_kernel.h"

// Map representation
#include "hesfm/semantic_map.h"

// Exploration module
#include "hesfm/exploration.h"

namespace hesfm {
/**
 * @brief Get HESFM version string
 */
inline std::string getVersionString() {
    return HESFM_VERSION_STRING;
}

/**
 * @brief Get HESFM version as a tuple 
 */
inline std::tuple<int, int, int> getVersion() {
    return {HESFM_VERSION_MAJOR, HESFM_VERSION_MINOR, HESFM_VERSION_PATCH};
}

/**
 * @brief HESFM Pipeline - Convenience wrapper for full processing pipeline
 * 
 * Encapsulates all processing stages in a single class for ease of use.
 * For more control, use individual components directly.
 */
class HESFMPipeline {
public:
    /**
     * @brief Constructor with configuration
     */
    explicit HESFMPipeline(const HESFMConfig& config = HESGMConfig())
        : config_(config),
          uncertainty_decom
}

} // namespace hesfm

#endif // HESFM_HESFM_H_