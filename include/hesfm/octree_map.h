21  /**
 * @file octree_map.h
 * @brief OcTree-based semantic map storage for efficient spatial queries
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * 
 * Provides hierarchical spatial indexing for semantic map cells
 * with efficient insertion, query, and raytracing operations.
 */

#ifndef HESFM_OCTREE_MAP_H
#define HESFM_OCTREE_MAP_H

#include "hesfm/types.h"
#include <memory>
#include <array>
#include <functional>

namespace hesfm {

/**
 * @brief OcTree node for hierarchical semantic map storage
 */
class OcTreeNode {
public:
    OcTreeNode();
    ~OcTreeNode() = default;
    
    /**
     * @brief Check if this is a leaf node
     */
    bool isLeaf() const { return !hasChildren(); }
    
    /**
     * @brief Check if node has any children
     */
    bool hasChildren() const;
    
    /**
     * @brief Get child node at index (0-7)
     */
    OcTreeNode* getChild(int index);
    const OcTreeNode* getChild(int index) const;
    
    /**
     * @brief Create child node at index
     */
    OcTreeNode* createChild(int index);
    
    /**
     * @brief Get/set semantic state
     */
    SemanticState& getState() { return state_; }
    const SemanticState& getState() const { return state_; }
    void setState(const SemanticState& state) { state_ = state; }
    
    /**
     * @brief Check if node has been observed
     */
    bool isOccupied() const { return state_.observation_count > 0; }
    
    /**
     * @brief Prune children if they are all identical
     */
    bool prune();

private:
    std::array<std::unique_ptr<OcTreeNode>, 8> children_;
    SemanticState state_;
};

/**
 * @brief OcTree-based semantic map
 */
class OcTreeMap {
public:
    struct Config {
        double resolution = 0.05;       // Leaf node size
        double origin_x = -10.0;
        double origin_y = -10.0;
        double origin_z = -0.5;
        double size_x = 20.0;
        double size_y = 20.0;
        double size_z = 3.0;
        int num_classes = 40;
        int max_depth = 16;             // Maximum tree depth
    };
    
    OcTreeMap(const Config& config = Config());
    ~OcTreeMap() = default;
    
    /**
     * @brief Insert or update semantic state at position
     */
    void updateNode(const Vector3d& position, const VectorXd& log_odds_update);
    
    /**
     * @brief Query semantic state at position
     * @return Optional state if node exists
     */
    std::optional<SemanticState> query(const Vector3d& position) const;
    
    /**
     * @brief Get predicted class at position
     * @return Class index or -1 if not observed
     */
    int getClass(const Vector3d& position) const;
    
    /**
     * @brief Raycast from origin to end point
     * @param origin Ray origin
     * @param end Ray end point
     * @param callback Called for each cell along ray
     * @return True if ray hit an occupied cell
     */
    bool raycast(const Vector3d& origin, const Vector3d& end,
                 std::function<bool(const Vector3d&, const SemanticState&)> callback) const;
    
    /**
     * @brief Get all leaf nodes
     */
    std::vector<std::pair<Vector3d, SemanticState>> getAllLeaves() const;
    
    /**
     * @brief Get nodes within bounding box
     */
    std::vector<std::pair<Vector3d, SemanticState>> getNodesInBBox(
        const Vector3d& min_pt, const Vector3d& max_pt) const;
    
    /**
     * @brief Clear all nodes
     */
    void clear();
    
    /**
     * @brief Prune tree to remove redundant nodes
     */
    void prune();
    
    /**
     * @brief Get memory usage in bytes
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Get number of leaf nodes
     */
    size_t getNumLeaves() const;
    
    /**
     * @brief Get configuration
     */
    const Config& getConfig() const { return config_; }

private:
    Config config_;
    std::unique_ptr<OcTreeNode> root_;
    size_t num_leaves_;
    
    /**
     * @brief Convert position to key (discrete coordinates)
     */
    std::tuple<int, int, int> positionToKey(const Vector3d& position) const;
    
    /**
     * @brief Convert key to position (cell center)
     */
    Vector3d keyToPosition(int kx, int ky, int kz) const;
    
    /**
     * @brief Get child index for position at given depth
     */
    int getChildIndex(const Vector3d& position, const Vector3d& center, double size) const;
    
    /**
     * @brief Recursive node search
     */
    OcTreeNode* searchNode(const Vector3d& position);
    const OcTreeNode* searchNode(const Vector3d& position) const;
    
    /**
     * @brief Recursive leaf collection
     */
    void collectLeaves(const OcTreeNode* node, const Vector3d& center, double size,
                       std::vector<std::pair<Vector3d, SemanticState>>& leaves) const;
};

} // namespace hesfm

#endif // HESFM_OCTREE_MAP_H
