/**
 * @file semantic_costmap_3d_node.cpp
 * @brief 3D Semantic Costmap for HESFM, inspired by SLIDE SLAM
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node generates a 3D semantic costmap that combines:
 * 1. Semantic class-based traversability costs
 * 2. Geometric traversability (slope, step height, roughness)
 * 3. Evidential uncertainty from HESFM semantic mapping
 * 
 * Inspired by:
 * - SLIDE SLAM (KumarRobotics): Object-level metric-semantic SLAM
 * - Learning-based Traversability Costmap: Semantic + geometric fusion
 * - STEPP: Terrain traversability estimation
 * 
 * Key Features:
 * - 3D voxel grid with semantic costs per voxel
 * - Height-layered 2D costmap projection for navigation
 * - Dynamic obstacle detection using semantic classes
 * - Uncertainty-aware cost computation
 * 
 * Subscriptions:
 *   - /hesfm/semantic_map (hesfm_msgs/SemanticMap): HESFM semantic voxel map
 *   - /semantic_cloud (sensor_msgs/PointCloud2): Semantic point cloud
 *   - /odom (nav_msgs/Odometry): Robot odometry
 * 
 * Publications:
 *   - /semantic_costmap_3d (visualization_msgs/MarkerArray): 3D costmap visualization
 *   - /semantic_costmap_2d (nav_msgs/OccupancyGrid): 2D projected costmap for move_base
 *   - /traversability_cloud (sensor_msgs/PointCloud2): Traversability-colored point cloud
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <unordered_map>
#include <mutex>
#include <cmath>

/**
 * @brief Semantic class traversability costs
 * 
 * Cost values: 0.0 = fully traversable, 1.0 = lethal obstacle
 * Based on SUN RGB-D 37-class labels
 */
struct SemanticTraversability {
    // Traversable surfaces (cost 0.0 - 0.2)
    static constexpr float FLOOR = 0.0f;
    static constexpr float FLOOR_MAT = 0.1f;
    
    // Low obstacles / uncertain (cost 0.3 - 0.5)
    static constexpr float CARPET = 0.2f;
    static constexpr float PAPER = 0.3f;
    static constexpr float TOWEL = 0.3f;
    static constexpr float CLOTHES = 0.4f;
    static constexpr float BAG = 0.4f;
    static constexpr float BOX = 0.5f;
    
    // Medium obstacles (cost 0.5 - 0.7)
    static constexpr float CHAIR = 0.6f;
    static constexpr float PILLOW = 0.5f;
    static constexpr float BOOKS = 0.5f;
    static constexpr float LAMP = 0.6f;
    
    // High obstacles (cost 0.7 - 0.9)
    static constexpr float TABLE = 0.7f;
    static constexpr float DESK = 0.7f;
    static constexpr float SOFA = 0.8f;
    static constexpr float BED = 0.8f;
    static constexpr float CABINET = 0.85f;
    static constexpr float SHELVES = 0.85f;
    static constexpr float BOOKSHELF = 0.85f;
    static constexpr float DRESSER = 0.85f;
    static constexpr float NIGHT_STAND = 0.8f;
    static constexpr float FRIDGE = 0.9f;
    static constexpr float TELEVISION = 0.8f;
    
    // Lethal obstacles (cost 0.95 - 1.0)
    static constexpr float WALL = 1.0f;
    static constexpr float DOOR = 0.95f;  // Could be open
    static constexpr float WINDOW = 1.0f;
    static constexpr float MIRROR = 1.0f;
    static constexpr float WHITEBOARD = 0.95f;
    static constexpr float TOILET = 0.95f;
    static constexpr float SINK = 0.95f;
    static constexpr float BATHTUB = 0.95f;
    
    // Dynamic obstacles (special handling)
    static constexpr float PERSON = 0.9f;  // High cost but not lethal
    
    // Overhead (not obstacle at ground level)
    static constexpr float CEILING = 0.0f;  // Ignored
    static constexpr float CURTAIN = 0.3f;
    static constexpr float BLINDS = 0.2f;
    static constexpr float PICTURE = 0.0f;  // On wall
    static constexpr float COUNTER = 0.7f;
    static constexpr float SHOWER_CURTAIN = 0.3f;
    
    // Default for unknown
    static constexpr float UNKNOWN = 0.5f;
};

/**
 * @brief 3D Voxel with semantic cost
 */
struct SemanticVoxel3D {
    float cost;              // Traversability cost [0, 1]
    float uncertainty;       // Epistemic uncertainty from HESFM
    uint8_t semantic_class;  // Dominant semantic class
    float height;            // Height above ground
    uint16_t observation_count;
    ros::Time last_seen;
    
    SemanticVoxel3D() : cost(0.5f), uncertainty(1.0f), semantic_class(0),
                        height(0.0f), observation_count(0) {}
};

/**
 * @brief Hash function for 3D voxel keys
 */
struct VoxelKeyHash {
    size_t operator()(const std::tuple<int, int, int>& key) const {
        auto h1 = std::hash<int>{}(std::get<0>(key));
        auto h2 = std::hash<int>{}(std::get<1>(key));
        auto h3 = std::hash<int>{}(std::get<2>(key));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

using VoxelKey = std::tuple<int, int, int>;
using VoxelMap = std::unordered_map<VoxelKey, SemanticVoxel3D, VoxelKeyHash>;

/**
 * @brief 3D Semantic Costmap Node
 */
class SemanticCostmap3DNode {
public:
    SemanticCostmap3DNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), tf_listener_(tf_buffer_) {
        
        loadParameters();
        initializeTraversabilityCosts();
        setupSubscribers();
        setupPublishers();
        
        // Timer for costmap publication
        publish_timer_ = nh_.createTimer(
            ros::Duration(1.0 / publish_rate_),
            &SemanticCostmap3DNode::publishCostmaps, this
        );
        
        ROS_INFO("Semantic Costmap 3D Node initialized");
        ROS_INFO("  Voxel size: %.3f m", voxel_size_);
        ROS_INFO("  Map size: %.1f x %.1f x %.1f m", 
                 map_size_x_, map_size_y_, map_size_z_);
        ROS_INFO("  Robot height: %.2f m", robot_height_);
    }

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        // Voxel grid parameters
        pnh_.param("voxel_size", voxel_size_, 0.1);  // 10cm voxels
        pnh_.param("map_size_x", map_size_x_, 20.0);
        pnh_.param("map_size_y", map_size_y_, 20.0);
        pnh_.param("map_size_z", map_size_z_, 3.0);
        
        // Robot parameters
        pnh_.param("robot_height", robot_height_, 0.5);
        pnh_.param("robot_radius", robot_radius_, 0.3);
        pnh_.param("ground_clearance", ground_clearance_, 0.05);
        
        // Cost computation
        pnh_.param("slope_cost_factor", slope_cost_factor_, 0.5);
        pnh_.param("step_cost_factor", step_cost_factor_, 0.3);
        pnh_.param("uncertainty_cost_factor", uncertainty_cost_factor_, 0.2);
        pnh_.param("lethal_cost_threshold", lethal_cost_threshold_, 0.9);
        
        // Frames
        pnh_.param<std::string>("map_frame", map_frame_, "map");
        pnh_.param<std::string>("robot_frame", robot_frame_, "base_link");
        
        // Publication
        pnh_.param("publish_rate", publish_rate_, 5.0);
        
        // Dataset
        pnh_.param("num_classes", num_classes_, 37);
    }
    
    void initializeTraversabilityCosts() {
        // Initialize cost lookup table for SUN RGB-D 37 classes
        class_costs_.resize(num_classes_, SemanticTraversability::UNKNOWN);
        
        // SUN RGB-D class indices
        class_costs_[0] = SemanticTraversability::WALL;
        class_costs_[1] = SemanticTraversability::FLOOR;
        class_costs_[2] = SemanticTraversability::CABINET;
        class_costs_[3] = SemanticTraversability::BED;
        class_costs_[4] = SemanticTraversability::CHAIR;
        class_costs_[5] = SemanticTraversability::SOFA;
        class_costs_[6] = SemanticTraversability::TABLE;
        class_costs_[7] = SemanticTraversability::DOOR;
        class_costs_[8] = SemanticTraversability::WINDOW;
        class_costs_[9] = SemanticTraversability::BOOKSHELF;
        class_costs_[10] = SemanticTraversability::PICTURE;
        class_costs_[11] = SemanticTraversability::COUNTER;
        class_costs_[12] = SemanticTraversability::BLINDS;
        class_costs_[13] = SemanticTraversability::DESK;
        class_costs_[14] = SemanticTraversability::SHELVES;
        class_costs_[15] = SemanticTraversability::CURTAIN;
        class_costs_[16] = SemanticTraversability::DRESSER;
        class_costs_[17] = SemanticTraversability::PILLOW;
        class_costs_[18] = SemanticTraversability::MIRROR;
        class_costs_[19] = SemanticTraversability::FLOOR_MAT;
        class_costs_[20] = SemanticTraversability::CLOTHES;
        class_costs_[21] = SemanticTraversability::CEILING;
        class_costs_[22] = SemanticTraversability::BOOKS;
        class_costs_[23] = SemanticTraversability::FRIDGE;
        class_costs_[24] = SemanticTraversability::TELEVISION;
        class_costs_[25] = SemanticTraversability::PAPER;
        class_costs_[26] = SemanticTraversability::TOWEL;
        class_costs_[27] = SemanticTraversability::SHOWER_CURTAIN;
        class_costs_[28] = SemanticTraversability::BOX;
        class_costs_[29] = SemanticTraversability::WHITEBOARD;
        class_costs_[30] = SemanticTraversability::PERSON;
        class_costs_[31] = SemanticTraversability::NIGHT_STAND;
        class_costs_[32] = SemanticTraversability::TOILET;
        class_costs_[33] = SemanticTraversability::SINK;
        class_costs_[34] = SemanticTraversability::LAMP;
        class_costs_[35] = SemanticTraversability::BATHTUB;
        class_costs_[36] = SemanticTraversability::BAG;
        
        ROS_INFO("Initialized traversability costs for %d classes", num_classes_);
    }
    
    void setupSubscribers() {
        // Semantic point cloud from semantic_cloud_node
        semantic_cloud_sub_ = nh_.subscribe(
            "semantic_cloud", 1,
            &SemanticCostmap3DNode::semanticCloudCallback, this
        );
        
        // Robot odometry
        odom_sub_ = nh_.subscribe(
            "odom", 1,
            &SemanticCostmap3DNode::odomCallback, this
        );
        
        ROS_INFO("Subscribed to semantic_cloud and odom topics");
    }
    
    void setupPublishers() {
        // 3D costmap visualization
        costmap_3d_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "semantic_costmap_3d", 1
        );
        
        // 2D projected costmap for move_base
        costmap_2d_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(
            "semantic_costmap_2d", 1
        );
        
        // Traversability-colored point cloud
        traversability_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
            "traversability_cloud", 1
        );
    }
    
    // =========================================================================
    // Callbacks
    // =========================================================================
    
    void semanticCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(voxel_mutex_);
        
        // Get transform to map frame
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform(
                map_frame_, msg->header.frame_id,
                msg->header.stamp, ros::Duration(0.1)
            );
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return;
        }
        
        // Iterate through points
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_label(*msg, "label");
        sensor_msgs::PointCloud2ConstIterator<float> iter_unc(*msg, "uncertainty");
        
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, 
                                        ++iter_label, ++iter_unc) {
            // Transform point to map frame
            geometry_msgs::Point pt_in, pt_out;
            pt_in.x = *iter_x;
            pt_in.y = *iter_y;
            pt_in.z = *iter_z;
            
            tf2::doTransform(pt_in, pt_out, transform);
            
            // Get voxel key
            VoxelKey key = worldToVoxel(pt_out.x, pt_out.y, pt_out.z);
            
            // Update voxel
            uint8_t label = static_cast<uint8_t>(*iter_label);
            float uncertainty = *iter_unc;
            
            updateVoxel(key, label, uncertainty, pt_out.z);
        }
    }
    
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        robot_pose_ = msg->pose.pose;
        has_odom_ = true;
    }
    
    // =========================================================================
    // Voxel Operations
    // =========================================================================
    
    VoxelKey worldToVoxel(double x, double y, double z) {
        int vx = static_cast<int>(std::floor(x / voxel_size_));
        int vy = static_cast<int>(std::floor(y / voxel_size_));
        int vz = static_cast<int>(std::floor(z / voxel_size_));
        return std::make_tuple(vx, vy, vz);
    }
    
    void voxelToWorld(const VoxelKey& key, double& x, double& y, double& z) {
        x = (std::get<0>(key) + 0.5) * voxel_size_;
        y = (std::get<1>(key) + 0.5) * voxel_size_;
        z = (std::get<2>(key) + 0.5) * voxel_size_;
    }
    
    void updateVoxel(const VoxelKey& key, uint8_t label, 
                     float uncertainty, double height) {
        auto& voxel = voxel_map_[key];
        
        // Get semantic cost
        float semantic_cost = (label < class_costs_.size()) 
                             ? class_costs_[label] 
                             : SemanticTraversability::UNKNOWN;
        
        // Exponential moving average update
        float alpha = 0.3f;
        if (voxel.observation_count == 0) {
            voxel.cost = semantic_cost;
            voxel.uncertainty = uncertainty;
            voxel.semantic_class = label;
            voxel.height = height;
        } else {
            voxel.cost = alpha * semantic_cost + (1.0f - alpha) * voxel.cost;
            voxel.uncertainty = alpha * uncertainty + (1.0f - alpha) * voxel.uncertainty;
            
            // Update semantic class if more confident
            if (uncertainty < voxel.uncertainty) {
                voxel.semantic_class = label;
            }
        }
        
        voxel.observation_count++;
        voxel.last_seen = ros::Time::now();
    }
    
    float computeTotalCost(const SemanticVoxel3D& voxel, double height_above_ground) {
        // Base semantic cost
        float cost = voxel.cost;
        
        // Add uncertainty penalty
        cost += uncertainty_cost_factor_ * voxel.uncertainty;
        
        // Height-based filtering
        // Objects above robot height are not obstacles
        if (height_above_ground > robot_height_) {
            // Ceiling, hanging objects - reduce cost
            cost *= 0.1f;
        }
        // Objects at ground level that are traversable
        else if (height_above_ground < ground_clearance_ && 
                 voxel.semantic_class == 1) {  // Floor
            cost = 0.0f;
        }
        
        return std::min(1.0f, std::max(0.0f, cost));
    }
    
    // =========================================================================
    // Publication
    // =========================================================================
    
    void publishCostmaps(const ros::TimerEvent&) {
        std::lock_guard<std::mutex> lock(voxel_mutex_);
        
        if (voxel_map_.empty()) {
            return;
        }
        
        // Get robot position for local map extraction
        double robot_x = 0.0, robot_y = 0.0, robot_z = 0.0;
        {
            std::lock_guard<std::mutex> odom_lock(odom_mutex_);
            if (has_odom_) {
                robot_x = robot_pose_.position.x;
                robot_y = robot_pose_.position.y;
                robot_z = robot_pose_.position.z;
            }
        }
        
        // Publish 3D visualization
        publish3DCostmap(robot_x, robot_y, robot_z);
        
        // Publish 2D projected costmap
        publish2DCostmap(robot_x, robot_y, robot_z);
        
        // Publish traversability point cloud
        publishTraversabilityCloud();
    }
    
    void publish3DCostmap(double robot_x, double robot_y, double robot_z) {
        visualization_msgs::MarkerArray markers;
        visualization_msgs::Marker marker;
        
        marker.header.frame_id = map_frame_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "semantic_costmap_3d";
        marker.type = visualization_msgs::Marker::CUBE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = voxel_size_ * 0.9;
        marker.scale.y = voxel_size_ * 0.9;
        marker.scale.z = voxel_size_ * 0.9;
        
        for (const auto& kv : voxel_map_) {
            double x, y, z;
            voxelToWorld(kv.first, x, y, z);
            
            // Only show voxels within local range
            double dx = x - robot_x;
            double dy = y - robot_y;
            if (dx*dx + dy*dy > map_size_x_*map_size_x_/4) {
                continue;
            }
            
            // Skip low-cost (traversable) voxels for clarity
            float cost = computeTotalCost(kv.second, z - robot_z);
            if (cost < 0.3f) {
                continue;
            }
            
            geometry_msgs::Point pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            marker.points.push_back(pt);
            
            // Color by cost: green (low) -> yellow -> red (high)
            std_msgs::ColorRGBA color;
            color.a = 0.7f;
            if (cost < 0.5f) {
                color.r = cost * 2.0f;
                color.g = 1.0f;
                color.b = 0.0f;
            } else {
                color.r = 1.0f;
                color.g = (1.0f - cost) * 2.0f;
                color.b = 0.0f;
            }
            marker.colors.push_back(color);
        }
        
        marker.id = 0;
        markers.markers.push_back(marker);
        costmap_3d_pub_.publish(markers);
    }
    
    void publish2DCostmap(double robot_x, double robot_y, double robot_z) {
        nav_msgs::OccupancyGrid grid;
        
        grid.header.frame_id = map_frame_;
        grid.header.stamp = ros::Time::now();
        
        // Grid parameters
        int grid_size = static_cast<int>(map_size_x_ / voxel_size_);
        grid.info.resolution = voxel_size_;
        grid.info.width = grid_size;
        grid.info.height = grid_size;
        grid.info.origin.position.x = robot_x - map_size_x_ / 2;
        grid.info.origin.position.y = robot_y - map_size_y_ / 2;
        grid.info.origin.position.z = 0;
        grid.info.origin.orientation.w = 1.0;
        
        // Initialize grid
        grid.data.resize(grid_size * grid_size, -1);  // Unknown
        
        // Project 3D voxels to 2D
        // For each (x, y) cell, take maximum cost within robot height range
        std::unordered_map<int, float> cell_max_cost;
        
        for (const auto& kv : voxel_map_) {
            double x, y, z;
            voxelToWorld(kv.first, x, y, z);
            
            // Check if within robot traversable height
            double height_above_ground = z - robot_z;
            if (height_above_ground < ground_clearance_ || 
                height_above_ground > robot_height_) {
                continue;  // Below ground or above robot
            }
            
            // Convert to grid cell
            int gx = static_cast<int>((x - grid.info.origin.position.x) / voxel_size_);
            int gy = static_cast<int>((y - grid.info.origin.position.y) / voxel_size_);
            
            if (gx < 0 || gx >= grid_size || gy < 0 || gy >= grid_size) {
                continue;
            }
            
            int cell_idx = gy * grid_size + gx;
            float cost = computeTotalCost(kv.second, height_above_ground);
            
            // Take maximum cost for this cell
            auto it = cell_max_cost.find(cell_idx);
            if (it == cell_max_cost.end() || cost > it->second) {
                cell_max_cost[cell_idx] = cost;
            }
        }
        
        // Fill grid
        for (const auto& kv : cell_max_cost) {
            // Convert cost [0, 1] to occupancy [0, 100]
            int occ = static_cast<int>(kv.second * 100);
            grid.data[kv.first] = std::min(100, std::max(0, occ));
        }
        
        costmap_2d_pub_.publish(grid);
    }
    
    void publishTraversabilityCloud() {
        if (traversability_cloud_pub_.getNumSubscribers() == 0) {
            return;
        }
        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        
        for (const auto& kv : voxel_map_) {
            double x, y, z;
            voxelToWorld(kv.first, x, y, z);
            
            float cost = computeTotalCost(kv.second, z);
            
            pcl::PointXYZRGB pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            
            // Color by traversability
            if (cost < 0.3f) {
                // Green - traversable
                pt.r = 0;
                pt.g = 255;
                pt.b = 0;
            } else if (cost < 0.6f) {
                // Yellow - caution
                pt.r = 255;
                pt.g = 255;
                pt.b = 0;
            } else if (cost < 0.9f) {
                // Orange - high cost
                pt.r = 255;
                pt.g = 128;
                pt.b = 0;
            } else {
                // Red - lethal
                pt.r = 255;
                pt.g = 0;
                pt.b = 0;
            }
            
            cloud.push_back(pt);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.frame_id = map_frame_;
        msg.header.stamp = ros::Time::now();
        
        traversability_cloud_pub_.publish(msg);
    }
    
    // =========================================================================
    // Member Variables
    // =========================================================================
    
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Subscribers
    ros::Subscriber semantic_cloud_sub_;
    ros::Subscriber odom_sub_;
    
    // Publishers
    ros::Publisher costmap_3d_pub_;
    ros::Publisher costmap_2d_pub_;
    ros::Publisher traversability_cloud_pub_;
    
    // Timer
    ros::Timer publish_timer_;
    
    // Voxel map
    VoxelMap voxel_map_;
    std::mutex voxel_mutex_;
    
    // Robot state
    geometry_msgs::Pose robot_pose_;
    std::mutex odom_mutex_;
    bool has_odom_ = false;
    
    // Parameters
    double voxel_size_;
    double map_size_x_, map_size_y_, map_size_z_;
    double robot_height_, robot_radius_, ground_clearance_;
    double slope_cost_factor_, step_cost_factor_, uncertainty_cost_factor_;
    double lethal_cost_threshold_;
    std::string map_frame_, robot_frame_;
    double publish_rate_;
    int num_classes_;
    
    // Traversability costs per class
    std::vector<float> class_costs_;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "semantic_costmap_3d_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    try {
        SemanticCostmap3DNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    
    return 0;
}
