/**
 * @file costmap_node.cpp
 * @brief ROS node for generating semantic-aware navigation costmaps
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node generates 2D occupancy grids from the 3D semantic map,
 * with semantic-aware cost assignment for navigation.
 * 
 * Subscriptions:
 *   - semantic_map (sensor_msgs/PointCloud2): 3D semantic map
 * 
 * Publications:
 *   - costmap (nav_msgs/OccupancyGrid): Standard navigation costmap
 *   - semantic_costmap (nav_msgs/OccupancyGrid): Semantic-weighted costmap
 *   - traversability_map (nav_msgs/OccupancyGrid): Binary traversability
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <unordered_map>
#include <mutex>

#include "hesfm/types.h"

/**
 * @brief Costmap Generator Node
 */
class CostmapNode {
public:
    CostmapNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), tf_listener_(tf_buffer_) {
        
        loadParameters();
        initializeCostmap();
        setupSubscribers();
        setupPublishers();
        setupServices();
        setupTimers();
        
        ROS_INFO("Costmap Node initialized");
        ROS_INFO("  Resolution: %.3f m", resolution_);
        ROS_INFO("  Size: %.1f x %.1f m", size_x_, size_y_);
        ROS_INFO("  Height range: [%.2f, %.2f] m", height_min_, height_max_);
    }

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        // Frame IDs
        pnh_.param<std::string>("map_frame", map_frame_, "map");
        pnh_.param<std::string>("robot_frame", robot_frame_, "base_link");
        
        // Map parameters
        pnh_.param("resolution", resolution_, 0.05);
        pnh_.param("size_x", size_x_, 20.0);
        pnh_.param("size_y", size_y_, 20.0);
        pnh_.param("origin_x", origin_x_, -10.0);
        pnh_.param("origin_y", origin_y_, -10.0);
        
        // Height range for projection
        pnh_.param("height_min", height_min_, 0.0);
        pnh_.param("height_max", height_max_, 0.5);
        
        // Inflation
        pnh_.param("inflation_radius", inflation_radius_, 0.3);
        pnh_.param("cost_scaling_factor", cost_scaling_factor_, 10.0);
        
        // Cost values
        pnh_.param("free_cost", free_cost_, 0);
        pnh_.param("unknown_cost", unknown_cost_, -1);
        pnh_.param("obstacle_cost", obstacle_cost_, 100);
        
        // Publish rate
        pnh_.param("publish_rate", publish_rate_, 5.0);
        
        // Traversable classes (floor, floor_mat)
        std::vector<int> trav_default = {1, 19};
        pnh_.param("traversable_classes", traversable_classes_, trav_default);
        
        // Obstacle classes
        std::vector<int> obs_default = {0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 23, 24, 30, 32, 33, 35};
        pnh_.param("obstacle_classes", obstacle_classes_, obs_default);
        
        // Semantic costs (per class)
        loadSemanticCosts();
    }
    
    void loadSemanticCosts() {
        // Default semantic costs [0-100]
        semantic_costs_.resize(40, 50);  // Default moderate cost
        
        // Free costs (traversable)
        for (int cls : traversable_classes_) {
            if (cls >= 0 && cls < 40) semantic_costs_[cls] = 0;
        }
        
        // High costs (obstacles)
        for (int cls : obstacle_classes_) {
            if (cls >= 0 && cls < 40) semantic_costs_[cls] = 100;
        }
        
        // Medium costs (furniture - can pass near but not through)
        semantic_costs_[2] = 80;   // cabinet
        semantic_costs_[6] = 80;   // table
        semantic_costs_[9] = 70;   // bookshelf
        semantic_costs_[11] = 70;  // counter
        semantic_costs_[14] = 60;  // shelves
        semantic_costs_[16] = 70;  // dresser
        semantic_costs_[28] = 40;  // box
        semantic_costs_[31] = 70;  // night_stand
        semantic_costs_[34] = 30;  // lamp
        semantic_costs_[36] = 30;  // bag
    }
    
    void initializeCostmap() {
        // Compute grid dimensions
        width_ = static_cast<int>(std::ceil(size_x_ / resolution_));
        height_ = static_cast<int>(std::ceil(size_y_ / resolution_));
        
        // Initialize costmap
        costmap_.resize(width_ * height_, unknown_cost_);
        semantic_costmap_.resize(width_ * height_, unknown_cost_);
        
        ROS_INFO("Costmap size: %d x %d cells", width_, height_);
    }
    
    void setupSubscribers() {
        map_sub_ = nh_.subscribe("semantic_map", 1,
                                  &CostmapNode::mapCallback, this);
    }
    
    void setupPublishers() {
        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("costmap", 1);
        semantic_costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(
            "semantic_costmap", 1);
        traversability_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(
            "traversability_map", 1);
    }
    
    void setupServices() {
        clear_srv_ = nh_.advertiseService("clear_costmap",
                                           &CostmapNode::clearCallback, this);
    }
    
    void setupTimers() {
        publish_timer_ = nh_.createTimer(
            ros::Duration(1.0 / publish_rate_),
            &CostmapNode::publishCallback, this);
    }
    
    // =========================================================================
    // Callbacks
    // =========================================================================
    
    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Reset costmaps
        std::fill(costmap_.begin(), costmap_.end(), unknown_cost_);
        std::fill(semantic_costmap_.begin(), semantic_costmap_.end(), unknown_cost_);
        
        // Find field indices
        int x_idx = -1, y_idx = -1, z_idx = -1;
        int label_idx = -1, conf_idx = -1;
        
        for (size_t i = 0; i < msg->fields.size(); ++i) {
            const auto& field = msg->fields[i];
            if (field.name == "x") x_idx = i;
            else if (field.name == "y") y_idx = i;
            else if (field.name == "z") z_idx = i;
            else if (field.name == "label") label_idx = i;
            else if (field.name == "confidence") conf_idx = i;
        }
        
        if (x_idx < 0 || y_idx < 0 || z_idx < 0) return;
        
        // Process points
        for (size_t i = 0; i < msg->width * msg->height; ++i) {
            const uint8_t* ptr = &msg->data[i * msg->point_step];
            
            float x, y, z;
            memcpy(&x, ptr + msg->fields[x_idx].offset, sizeof(float));
            memcpy(&y, ptr + msg->fields[y_idx].offset, sizeof(float));
            memcpy(&z, ptr + msg->fields[z_idx].offset, sizeof(float));
            
            // Filter by height
            if (z < height_min_ || z > height_max_) continue;
            
            // Convert to grid coordinates
            int gx = static_cast<int>((x - origin_x_) / resolution_);
            int gy = static_cast<int>((y - origin_y_) / resolution_);
            
            if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) continue;
            
            int idx = gy * width_ + gx;
            
            // Get label and confidence
            int label = 0;
            float confidence = 0.8f;
            
            if (label_idx >= 0) {
                uint32_t lbl;
                memcpy(&lbl, ptr + msg->fields[label_idx].offset, sizeof(uint32_t));
                label = static_cast<int>(lbl);
            }
            
            if (conf_idx >= 0) {
                memcpy(&confidence, ptr + msg->fields[conf_idx].offset, sizeof(float));
            }
            
            // Assign costs
            updateCellCost(idx, label, confidence);
        }
        
        // Apply inflation
        applyInflation();
        
        last_update_ = ros::Time::now();
    }
    
    bool clearCallback(std_srvs::Empty::Request& req,
                       std_srvs::Empty::Response& res) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::fill(costmap_.begin(), costmap_.end(), unknown_cost_);
        std::fill(semantic_costmap_.begin(), semantic_costmap_.end(), unknown_cost_);
        ROS_INFO("Costmap cleared");
        return true;
    }
    
    void publishCallback(const ros::TimerEvent&) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        ros::Time now = ros::Time::now();
        
        // Standard costmap
        if (costmap_pub_.getNumSubscribers() > 0) {
            nav_msgs::OccupancyGrid msg;
            fillOccupancyGridMsg(msg, costmap_, now);
            costmap_pub_.publish(msg);
        }
        
        // Semantic costmap
        if (semantic_costmap_pub_.getNumSubscribers() > 0) {
            nav_msgs::OccupancyGrid msg;
            fillOccupancyGridMsg(msg, semantic_costmap_, now);
            semantic_costmap_pub_.publish(msg);
        }
        
        // Traversability map
        if (traversability_pub_.getNumSubscribers() > 0) {
            std::vector<int8_t> trav_map(costmap_.size());
            for (size_t i = 0; i < costmap_.size(); ++i) {
                if (costmap_[i] == unknown_cost_) {
                    trav_map[i] = -1;
                } else if (costmap_[i] < 50) {
                    trav_map[i] = 0;
                } else {
                    trav_map[i] = 100;
                }
            }
            
            nav_msgs::OccupancyGrid msg;
            fillOccupancyGridMsg(msg, trav_map, now);
            traversability_pub_.publish(msg);
        }
    }
    
    // =========================================================================
    // Cost Computation
    // =========================================================================
    
    void updateCellCost(int idx, int label, float confidence) {
        // Standard binary costmap
        bool is_traversable = std::find(traversable_classes_.begin(),
                                         traversable_classes_.end(),
                                         label) != traversable_classes_.end();
        bool is_obstacle = std::find(obstacle_classes_.begin(),
                                      obstacle_classes_.end(),
                                      label) != obstacle_classes_.end();
        
        if (is_traversable && confidence > 0.3) {
            costmap_[idx] = free_cost_;
        } else if (is_obstacle && confidence > 0.3) {
            costmap_[idx] = obstacle_cost_;
        } else if (costmap_[idx] == unknown_cost_) {
            // Mark as partially known
            costmap_[idx] = 50;
        }
        
        // Semantic costmap with per-class costs
        if (label >= 0 && label < static_cast<int>(semantic_costs_.size())) {
            int sem_cost = semantic_costs_[label];
            
            // Weight by confidence
            sem_cost = static_cast<int>(sem_cost * confidence + 50 * (1.0f - confidence));
            
            // Take max cost (conservative)
            if (semantic_costmap_[idx] == unknown_cost_ || sem_cost > semantic_costmap_[idx]) {
                semantic_costmap_[idx] = sem_cost;
            }
        }
    }
    
    void applyInflation() {
        if (inflation_radius_ <= 0) return;
        
        int inflation_cells = static_cast<int>(std::ceil(inflation_radius_ / resolution_));
        
        std::vector<int8_t> inflated_costmap = costmap_;
        
        for (int cy = 0; cy < height_; ++cy) {
            for (int cx = 0; cx < width_; ++cx) {
                int cidx = cy * width_ + cx;
                
                if (costmap_[cidx] != obstacle_cost_) continue;
                
                // Inflate around this obstacle cell
                for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
                    for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
                        int nx = cx + dx;
                        int ny = cy + dy;
                        
                        if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
                        
                        int nidx = ny * width_ + nx;
                        
                        // Don't inflate into unknown or already obstacle
                        if (costmap_[nidx] == unknown_cost_ || 
                            costmap_[nidx] == obstacle_cost_) continue;
                        
                        // Compute distance
                        double dist = std::sqrt(dx*dx + dy*dy) * resolution_;
                        
                        if (dist >= inflation_radius_) continue;
                        
                        // Exponential decay cost
                        double decay = std::exp(-cost_scaling_factor_ * 
                                               (dist / inflation_radius_));
                        int inf_cost = static_cast<int>(obstacle_cost_ * decay);
                        
                        inflated_costmap[nidx] = std::max(inflated_costmap[nidx],
                                                           static_cast<int8_t>(inf_cost));
                    }
                }
            }
        }
        
        costmap_ = inflated_costmap;
    }
    
    void fillOccupancyGridMsg(nav_msgs::OccupancyGrid& msg,
                               const std::vector<int8_t>& data,
                               const ros::Time& stamp) {
        msg.header.stamp = stamp;
        msg.header.frame_id = map_frame_;
        
        msg.info.resolution = resolution_;
        msg.info.width = width_;
        msg.info.height = height_;
        msg.info.origin.position.x = origin_x_;
        msg.info.origin.position.y = origin_y_;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;
        
        msg.data = data;
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
    ros::Subscriber map_sub_;
    
    // Publishers
    ros::Publisher costmap_pub_;
    ros::Publisher semantic_costmap_pub_;
    ros::Publisher traversability_pub_;
    
    // Services
    ros::ServiceServer clear_srv_;
    
    // Timer
    ros::Timer publish_timer_;
    
    // Costmap data
    std::vector<int8_t> costmap_;
    std::vector<int8_t> semantic_costmap_;
    std::mutex mutex_;
    
    // Parameters
    std::string map_frame_;
    std::string robot_frame_;
    double resolution_;
    double size_x_, size_y_;
    double origin_x_, origin_y_;
    double height_min_, height_max_;
    double inflation_radius_;
    double cost_scaling_factor_;
    int free_cost_, unknown_cost_, obstacle_cost_;
    double publish_rate_;
    int width_, height_;
    
    std::vector<int> traversable_classes_;
    std::vector<int> obstacle_classes_;
    std::vector<int> semantic_costs_;
    
    ros::Time last_update_;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "costmap_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    try {
        CostmapNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    
    return 0;
}
