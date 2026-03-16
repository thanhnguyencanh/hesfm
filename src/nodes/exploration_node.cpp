/**
 * @file exploration_node.cpp
 * @brief ROS node for EMI-based exploration goal generation
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node generates exploration goals based on Extended Mutual Information
 * that considers both geometric and semantic uncertainty.
 * 
 * Subscriptions:
 *   - semantic_map (sensor_msgs/PointCloud2): Current semantic map
 *   - robot_pose (geometry_msgs/PoseStamped): Current robot pose
 * 
 * Publications:
 *   - exploration_goals (geometry_msgs/PoseArray): Ranked exploration goals
 *   - exploration_markers (visualization_msgs/MarkerArray): Goal visualization
 *   - frontiers (visualization_msgs/MarkerArray): Detected frontiers
 * 
 * Services:
 *   - get_exploration_goal (hesfm/GetExplorationGoals): Get ranked goals
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/ColorRGBA.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "hesfm/hesfm.h"

/**
 * @brief Exploration Node using EMI-based goal selection
 */
class ExplorationNode {
public:
    ExplorationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), tf_listener_(tf_buffer_) {
        
        loadParameters();
        initializeExploration();
        setupSubscribers();
        setupPublishers();
        setupTimers();
        
        ROS_INFO("Exploration Node initialized");
        ROS_INFO("  Map frame: %s", map_frame_.c_str());
        ROS_INFO("  Robot frame: %s", robot_frame_.c_str());
        ROS_INFO("  Sensor range: %.2f m", config_.sensor_range);
    }

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        // Frame IDs
        pnh_.param<std::string>("map_frame", map_frame_, "map");
        pnh_.param<std::string>("robot_frame", robot_frame_, "base_link");
        
        // Exploration parameters
        pnh_.param("sensor_range", config_.sensor_range, 6.0);
        pnh_.param("sensor_fov_horizontal", config_.sensor_fov_horizontal, 1.2);
        pnh_.param("sensor_fov_vertical", config_.sensor_fov_vertical, 0.9);
        pnh_.param("max_distance", config_.max_distance, 10.0);
        pnh_.param("min_info_gain", config_.min_info_gain, 0.1);
        pnh_.param("max_goals", config_.max_goals, 10);
        pnh_.param("min_obstacle_distance", config_.min_obstacle_distance, 0.5);
        pnh_.param("min_frontier_size", config_.min_frontier_size, 10);
        
        // Utility weights
        pnh_.param("weight_info_gain", config_.weight_info_gain, 1.0);
        pnh_.param("weight_distance", config_.weight_distance, 0.3);
        pnh_.param("weight_uncertainty", config_.weight_uncertainty, 0.5);
        
        // Update rate
        pnh_.param("update_rate", update_rate_, 1.0);
        
        // Auto-navigation
        pnh_.param("auto_navigate", auto_navigate_, false);
    }
    
    void initializeExploration() {
        planner_ = std::make_unique<hesfm::ExplorationPlanner>(config_);
        
        // Initialize map
        hesfm::MapConfig map_config;
        pnh_.param("resolution", map_config.resolution, 0.05);
        pnh_.param("num_classes", map_config.num_classes, 40);
        semantic_map_ = std::make_unique<hesfm::SemanticMap>(map_config);
        
        ROS_INFO("Exploration planner initialized");
    }
    
    void setupSubscribers() {
        // Semantic map subscriber
        map_sub_ = nh_.subscribe("semantic_map", 1,
                                  &ExplorationNode::mapCallback, this);
        
        // Robot pose subscriber (optional, can also use TF)
        pose_sub_ = nh_.subscribe("robot_pose", 1,
                                   &ExplorationNode::poseCallback, this);
    }
    
    void setupPublishers() {
        // Exploration goals as PoseArray
        goals_pub_ = nh_.advertise<geometry_msgs::PoseArray>("exploration_goals", 1);
        
        // Visualization markers
        goal_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "exploration_markers", 1);
        frontier_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "frontiers", 1);
        
        // Best goal for navigation
        best_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(
            "exploration_goal", 1);
    }
    
    void setupTimers() {
        // Exploration update timer
        update_timer_ = nh_.createTimer(
            ros::Duration(1.0 / update_rate_),
            &ExplorationNode::updateCallback, this);
    }
    
    // =========================================================================
    // Callbacks
    // =========================================================================
    
    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        // Update internal map representation
        updateMapFromPointCloud(msg);
        has_map_ = true;
    }
    
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        robot_position_ = hesfm::Vector3d(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z);
        
        robot_orientation_ = hesfm::Quaterniond(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);
        
        has_pose_ = true;
    }
    
    void updateCallback(const ros::TimerEvent&) {
        // Get robot pose from TF if not provided via topic
        if (!has_pose_) {
            if (!getRobotPoseFromTF()) {
                ROS_WARN_THROTTLE(5.0, "Cannot get robot pose");
                return;
            }
        }
        
        if (!has_map_) {
            ROS_WARN_THROTTLE(5.0, "Waiting for map...");
            return;
        }
        
        // Compute exploration goals
        auto goals = planner_->computeGoals(
            *semantic_map_, robot_position_, robot_orientation_);
        
        if (goals.empty()) {
            ROS_INFO_THROTTLE(10.0, "No exploration goals found - exploration may be complete");
            
            if (planner_->isExplorationComplete(*semantic_map_)) {
                ROS_INFO("Exploration complete!");
            }
            return;
        }
        
        // Publish goals
        publishGoals(goals);
        publishGoalMarkers(goals);
        publishFrontierMarkers();
        
        // Publish best goal
        publishBestGoal(goals[0]);
        
        ROS_DEBUG("Published %zu exploration goals, best EMI: %.3f",
                  goals.size(), goals[0].emi_value);
    }
    
    // =========================================================================
    // Map Update
    // =========================================================================
    
    void updateMapFromPointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        // Parse point cloud and update internal map
        // This is a simplified version - full implementation would use
        // the actual HESFM map update
        
        const auto& config = semantic_map_->getConfig();
        
        // Iterate through points
        for (size_t i = 0; i < msg->width * msg->height; ++i) {
            const uint8_t* ptr = &msg->data[i * msg->point_step];
            
            float x, y, z;
            memcpy(&x, ptr + 0, sizeof(float));
            memcpy(&y, ptr + 4, sizeof(float));
            memcpy(&z, ptr + 8, sizeof(float));
            
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }
            
            // Get label if available (assumes specific field layout)
            uint32_t label = 0;
            if (msg->point_step >= 20) {
                memcpy(&label, ptr + 16, sizeof(uint32_t));
            }
            
            // Create simple probability distribution
            std::vector<double> probs(config.num_classes, 0.01);
            if (label < static_cast<uint32_t>(config.num_classes)) {
                probs[label] = 0.9;
            }
            
            // Update cell
            semantic_map_->updateCell(hesfm::Vector3d(x, y, z), probs, 1.0);
        }
    }
    
    // =========================================================================
    // TF
    // =========================================================================
    
    bool getRobotPoseFromTF() {
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                map_frame_, robot_frame_, ros::Time(0), ros::Duration(0.1));
            
            robot_position_ = hesfm::Vector3d(
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z);
            
            robot_orientation_ = hesfm::Quaterniond(
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z);
            
            return true;
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(5.0, "TF lookup failed: %s", ex.what());
            return false;
        }
    }
    
    // =========================================================================
    // Publishing
    // =========================================================================
    
    void publishGoals(const std::vector<hesfm::ExplorationGoal>& goals) {
        geometry_msgs::PoseArray msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        
        for (const auto& goal : goals) {
            geometry_msgs::Pose pose;
            pose.position.x = goal.position.x();
            pose.position.y = goal.position.y();
            pose.position.z = goal.position.z();
            pose.orientation.x = goal.orientation.x();
            pose.orientation.y = goal.orientation.y();
            pose.orientation.z = goal.orientation.z();
            pose.orientation.w = goal.orientation.w();
            
            msg.poses.push_back(pose);
        }
        
        goals_pub_.publish(msg);
    }
    
    void publishBestGoal(const hesfm::ExplorationGoal& goal) {
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        
        msg.pose.position.x = goal.position.x();
        msg.pose.position.y = goal.position.y();
        msg.pose.position.z = goal.position.z();
        msg.pose.orientation.x = goal.orientation.x();
        msg.pose.orientation.y = goal.orientation.y();
        msg.pose.orientation.z = goal.orientation.z();
        msg.pose.orientation.w = goal.orientation.w();
        
        best_goal_pub_.publish(msg);
    }
    
    void publishGoalMarkers(const std::vector<hesfm::ExplorationGoal>& goals) {
        visualization_msgs::MarkerArray markers;
        
        // Delete all previous markers
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        markers.markers.push_back(delete_marker);
        
        for (size_t i = 0; i < goals.size(); ++i) {
            const auto& goal = goals[i];
            
            // Goal position marker (arrow)
            visualization_msgs::Marker arrow;
            arrow.header.frame_id = map_frame_;
            arrow.header.stamp = ros::Time::now();
            arrow.ns = "exploration_goals";
            arrow.id = static_cast<int>(i);
            arrow.type = visualization_msgs::Marker::ARROW;
            arrow.action = visualization_msgs::Marker::ADD;
            
            arrow.pose.position.x = goal.position.x();
            arrow.pose.position.y = goal.position.y();
            arrow.pose.position.z = goal.position.z();
            arrow.pose.orientation.x = goal.orientation.x();
            arrow.pose.orientation.y = goal.orientation.y();
            arrow.pose.orientation.z = goal.orientation.z();
            arrow.pose.orientation.w = goal.orientation.w();
            
            arrow.scale.x = 0.5;  // Arrow length
            arrow.scale.y = 0.1;  // Arrow width
            arrow.scale.z = 0.1;  // Arrow height
            
            // Color by rank (green = best, red = worst)
            float rank_ratio = static_cast<float>(i) / static_cast<float>(goals.size());
            arrow.color.r = rank_ratio;
            arrow.color.g = 1.0f - rank_ratio;
            arrow.color.b = 0.0f;
            arrow.color.a = 0.8f;
            
            arrow.lifetime = ros::Duration(1.0 / update_rate_ + 0.1);
            
            markers.markers.push_back(arrow);
            
            // EMI value text
            visualization_msgs::Marker text;
            text.header = arrow.header;
            text.ns = "exploration_emi";
            text.id = static_cast<int>(i);
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD;
            
            text.pose.position.x = goal.position.x();
            text.pose.position.y = goal.position.y();
            text.pose.position.z = goal.position.z() + 0.5;
            
            text.scale.z = 0.2;
            text.color.r = text.color.g = text.color.b = 1.0;
            text.color.a = 1.0;
            
            char buf[32];
            snprintf(buf, sizeof(buf), "#%d: %.2f", goal.rank, goal.emi_value);
            text.text = buf;
            
            text.lifetime = ros::Duration(1.0 / update_rate_ + 0.1);
            
            markers.markers.push_back(text);
        }
        
        goal_markers_pub_.publish(markers);
    }
    
    void publishFrontierMarkers() {
        auto frontiers = planner_->detectFrontiers(*semantic_map_);
        
        visualization_msgs::MarkerArray markers;
        
        // Delete all previous markers
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        markers.markers.push_back(delete_marker);
        
        for (size_t i = 0; i < frontiers.size(); ++i) {
            const auto& frontier = frontiers[i];
            
            // Frontier cells as points
            visualization_msgs::Marker points;
            points.header.frame_id = map_frame_;
            points.header.stamp = ros::Time::now();
            points.ns = "frontier_cells";
            points.id = static_cast<int>(i);
            points.type = visualization_msgs::Marker::POINTS;
            points.action = visualization_msgs::Marker::ADD;
            
            points.scale.x = 0.05;
            points.scale.y = 0.05;
            
            // Color by frontier ID
            points.color.r = static_cast<float>((frontier.id * 37) % 256) / 255.0f;
            points.color.g = static_cast<float>((frontier.id * 91) % 256) / 255.0f;
            points.color.b = static_cast<float>((frontier.id * 157) % 256) / 255.0f;
            points.color.a = 0.8f;
            
            for (const auto& cell : frontier.cells) {
                geometry_msgs::Point p;
                p.x = cell.position.x();
                p.y = cell.position.y();
                p.z = cell.position.z();
                points.points.push_back(p);
            }
            
            points.lifetime = ros::Duration(1.0 / update_rate_ + 0.1);
            
            markers.markers.push_back(points);
            
            // Frontier centroid
            visualization_msgs::Marker centroid;
            centroid.header = points.header;
            centroid.ns = "frontier_centroids";
            centroid.id = static_cast<int>(i);
            centroid.type = visualization_msgs::Marker::SPHERE;
            centroid.action = visualization_msgs::Marker::ADD;
            
            centroid.pose.position.x = frontier.centroid.x();
            centroid.pose.position.y = frontier.centroid.y();
            centroid.pose.position.z = frontier.centroid.z();
            centroid.pose.orientation.w = 1.0;
            
            centroid.scale.x = centroid.scale.y = centroid.scale.z = 0.2;
            centroid.color = points.color;
            centroid.color.a = 1.0;
            
            centroid.lifetime = ros::Duration(1.0 / update_rate_ + 0.1);
            
            markers.markers.push_back(centroid);
        }
        
        frontier_markers_pub_.publish(markers);
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
    ros::Subscriber pose_sub_;
    
    // Publishers
    ros::Publisher goals_pub_;
    ros::Publisher best_goal_pub_;
    ros::Publisher goal_markers_pub_;
    ros::Publisher frontier_markers_pub_;
    
    // Timer
    ros::Timer update_timer_;
    
    // Exploration
    hesfm::ExplorationConfig config_;
    std::unique_ptr<hesfm::ExplorationPlanner> planner_;
    std::unique_ptr<hesfm::SemanticMap> semantic_map_;
    
    // Robot state
    hesfm::Vector3d robot_position_ = hesfm::Vector3d::Zero();
    hesfm::Quaterniond robot_orientation_ = hesfm::Quaterniond::Identity();
    
    // Parameters
    std::string map_frame_;
    std::string robot_frame_;
    double update_rate_;
    bool auto_navigate_;
    
    // State flags
    bool has_map_ = false;
    bool has_pose_ = false;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "exploration_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    try {
        ExplorationNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    
    return 0;
}
