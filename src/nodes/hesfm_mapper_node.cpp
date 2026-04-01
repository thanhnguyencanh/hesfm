/**
 * @file hesfm_mapper_node.cpp
 * @brief Main ROS node for HESFM semantic mapping
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node subscribes to semantic point clouds and maintains
 * the semantic map using the HESFM framework.
 * 
 * 
 * 
 * Subscriptions:
 *   - semantic_cloud (sensor_msgs/PointCloud2): Semantic point cloud
 * 
 * Publications:
 *   - semantic_map (sensor_msgs/PointCloud2): 3D semantic map
 *   - costmap (nav_msgs/OccupancyGrid): 2D navigation costmap
 *   - primitives (visualization_msgs/MarkerArray): Gaussian primitives
 *   - uncertainty_info (hesfm/UncertaintyInfo): Uncertainty statistics
 *   - semantic_costmap_3d (visualization_msgs/MarkerArray): 3D semantic costmap [NEW]
 *   - semantic_costmap_2d (nav_msgs/OccupancyGrid): 2D semantic costmap [NEW]
 *   - traversability_cloud (sensor_msgs/PointCloud2): Traversability visualization [NEW]
 *
 * Services:
 *   - get_semantic_map (hesfm/GetSemanticMap)
 *   - query_point (hesfm/QueryPoint)
 *   - reset_map (hesfm/ResetMap)
 *   - save_map (hesfm/SaveMap)
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <dynamic_reconfigure/server.h>

#include "hesfm/hesfm.h"

// Custom point type for semantic point cloud
struct PointXYZRGBLU {
    PCL_ADD_POINT4D;
    PCL_ADD_RGB;
    uint32_t label;
    float uncertainty;
    float probabilities[40];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBLU,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, rgb, rgb)
    (uint32_t, label, label)
    (float, uncertainty, uncertainty)
)

/**
 * @brief HESFM Mapper ROS Node
 */
class HESFMMapperNode {
public:
    HESFMMapperNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), tf_listener_(tf_buffer_) {
        
        // Load parameters
        loadParameters();
        
        // Initialize HESFM components
        initializeHESFM();
        
        // Setup ROS interfaces
        setupSubscribers();
        setupPublishers();
        setupServices();
        setupTimers();
        
        ROS_INFO("HESFM Mapper Node initialized");
        ROS_INFO("  Map frame: %s", map_frame_.c_str());
        ROS_INFO("  Resolution: %.3f m", config_.map.resolution);
        ROS_INFO("  Num classes: %d", config_.map.num_classes);
    }
    
    ~HESFMMapperNode() = default;

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        // Frame IDs
        pnh_.param<std::string>("map_frame", map_frame_, "map");
        pnh_.param<std::string>("sensor_frame", sensor_frame_, "camera_color_optical_frame");

        // Dataset for class colors
        std::string dataset;
        pnh_.param<std::string>("esanet_dataset", dataset, "sunrgbd");
        class_colors_ = (dataset == "nyuv2") ? hesfm::NYUV2_CLASS_COLORS
                                              : hesfm::SUNRGBD_CLASS_COLORS;

        // Map parameters — YAML nested under "map/"
        pnh_.param("map/resolution",   config_.map.resolution,   0.05);
        pnh_.param("map/num_classes",  config_.map.num_classes,   37);
        pnh_.param("map/size_x",       config_.map.size_x,       20.0);
        pnh_.param("map/size_y",       config_.map.size_y,       20.0);
        pnh_.param("map/size_z",       config_.map.size_z,        3.0);
        pnh_.param("map/origin_x",     config_.map.origin_x,    -10.0);
        pnh_.param("map/origin_y",     config_.map.origin_y,    -10.0);
        pnh_.param("map/origin_z",     config_.map.origin_z,     -0.5);
        // Allow flat overrides from launch args
        pnh_.param("resolution",       config_.map.resolution,   config_.map.resolution);
        pnh_.param("num_classes",      config_.map.num_classes,  config_.map.num_classes);
        pnh_.param("map_size_x",       config_.map.size_x,       config_.map.size_x);
        pnh_.param("map_size_y",       config_.map.size_y,       config_.map.size_y);
        pnh_.param("map_size_z",       config_.map.size_z,       config_.map.size_z);

        // Uncertainty weights — YAML nested under "uncertainty/"
        pnh_.param("uncertainty/w_semantic",    config_.uncertainty.w_semantic,    0.40);
        pnh_.param("uncertainty/w_spatial",     config_.uncertainty.w_spatial,     0.20);
        pnh_.param("uncertainty/w_observation", config_.uncertainty.w_observation, 0.25);
        pnh_.param("uncertainty/w_temporal",    config_.uncertainty.w_temporal,    0.15);

        // Gaussian primitive parameters — YAML nested under "primitives/"
        pnh_.param("primitives/target_primitives",        config_.primitive.target_primitives,        512);
        pnh_.param("primitives/min_points_per_primitive", config_.primitive.min_points_per_primitive,  5);
        pnh_.param("primitives/regularization",           config_.primitive.regularization,            0.01);
        // High conflict_threshold keeps most primitives (low = too aggressive filtering)
        pnh_.param("primitives/conflict_threshold",       config_.primitive.conflict_threshold,        0.9);

        // Kernel parameters — YAML nested under "kernel/"
        pnh_.param("kernel/length_scale_min",      config_.kernel.length_scale_min,      0.1);
        pnh_.param("kernel/length_scale_max",      config_.kernel.length_scale_max,      0.5);
        // High uncertainty_threshold keeps high-uncertainty primitives too
        pnh_.param("kernel/uncertainty_threshold", config_.kernel.uncertainty_threshold,  0.95);
        pnh_.param("kernel/gamma",                 config_.kernel.gamma,                  2.0);

        // Navigation parameters — YAML nested under "navigation/"
        pnh_.param("navigation/costmap_height_min", config_.navigation.costmap_height_min, 0.0);
        pnh_.param("navigation/costmap_height_max", config_.navigation.costmap_height_max, 0.5);

        // Processing parameters — YAML nested under "processing/"
        pnh_.param("processing/map_publish_rate",     config_.processing.map_publish_rate,     2.0);
        pnh_.param("processing/costmap_publish_rate", config_.processing.costmap_publish_rate, 5.0);

        // Traversable classes
        std::vector<int> traversable_default = {1, 19};  // floor, floor_mat (SUNRGBD)
        pnh_.param("navigation/traversable_classes", traversable_classes_, traversable_default);
    }
    
    void initializeHESFM() {
        // Create HESFM pipeline
        pipeline_ = std::make_unique<hesfm::HESFMPipeline>(config_);
        
        // Set traversable classes
        std::set<int> trav_set(traversable_classes_.begin(), traversable_classes_.end());
        pipeline_->getMap().setTraversableClasses(trav_set);
        pipeline_->getKernel().setTraversableClasses(trav_set);
        
        ROS_INFO("HESFM pipeline initialized");
    }
    
    void setupSubscribers() {
        // Semantic point cloud subscriber
        cloud_sub_ = nh_.subscribe("semantic_cloud", 1, 
                                    &HESFMMapperNode::cloudCallback, this);
        
        ROS_INFO("Subscribed to: %s", cloud_sub_.getTopic().c_str());
    }
    
    void setupPublishers() {
        // Semantic map publisher
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_map", 1);
        
        // Costmap publisher
        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("costmap", 1);
        
        // Primitives visualization publisher
        primitives_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("primitives", 1);
        
        // Uncertainty map publisher
        uncertainty_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("uncertainty_map", 1);
        
        ROS_INFO("Publishers initialized");
    }
    
    void setupServices() {
        // Reset map service
        reset_srv_ = nh_.advertiseService("reset_map", 
                                           &HESFMMapperNode::resetMapCallback, this);
        
        ROS_INFO("Services initialized");
    }
    
    void setupTimers() {
        // Map publishing timer
        map_timer_ = nh_.createTimer(
            ros::Duration(1.0 / config_.processing.map_publish_rate),
            &HESFMMapperNode::mapPublishCallback, this);
        
        // Costmap publishing timer
        costmap_timer_ = nh_.createTimer(
            ros::Duration(1.0 / config_.processing.costmap_publish_rate),
            &HESFMMapperNode::costmapPublishCallback, this);
    }
    
    // =========================================================================
    // Callbacks
    // =========================================================================
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        ros::Time start_time = ros::Time::now();
        
        // Get transform from sensor to map frame
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform(
                map_frame_, msg->header.frame_id,
                msg->header.stamp, ros::Duration(0.1));
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return;
        }
        
        // Extract sensor origin
        hesfm::Vector3d sensor_origin(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z);
        
        // Convert point cloud to HESFM format
        std::vector<hesfm::SemanticPoint> points;
        if (!convertPointCloud(msg, transform, points)) {
            ROS_WARN_THROTTLE(1.0, "Failed to convert point cloud");
            return;
        }
        
        if (points.empty()) {
            return;
        }
        
        // Process through HESFM pipeline
        int num_primitives = pipeline_->process(points, sensor_origin);
        
        // Store current primitives for visualization
        current_primitives_ = pipeline_->getPrimitiveBuilder().buildPrimitives(points);
        
        // Update timing statistics
        double processing_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        updateStatistics(processing_time, points.size(), num_primitives);
        
        ROS_DEBUG("Processed %zu points, %d primitives in %.1f ms",
                  points.size(), num_primitives, processing_time);
    }
    
    void mapPublishCallback(const ros::TimerEvent&) {
        publishSemanticMap();
        publishPrimitives();
        publishUncertaintyMap();
    }
    
    void costmapPublishCallback(const ros::TimerEvent&) {
        publishCostmap();
    }
    
    bool resetMapCallback(std_srvs::Empty::Request& req,
                          std_srvs::Empty::Response& res) {
        pipeline_->reset();
        ROS_INFO("Map reset");
        return true;
    }
    
    // =========================================================================
    // Point Cloud Conversion
    // =========================================================================
    
    bool convertPointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg,
                           const geometry_msgs::TransformStamped& transform,
                           std::vector<hesfm::SemanticPoint>& points) {
        
        // Check for required fields
        int x_idx = -1, y_idx = -1, z_idx = -1;
        int label_idx = -1, uncertainty_idx = -1, rgb_idx = -1;
        
        for (size_t i = 0; i < msg->fields.size(); ++i) {
            const auto& field = msg->fields[i];
            if (field.name == "x") x_idx = i;
            else if (field.name == "y") y_idx = i;
            else if (field.name == "z") z_idx = i;
            else if (field.name == "label") label_idx = i;
            else if (field.name == "uncertainty") uncertainty_idx = i;
            else if (field.name == "rgb") rgb_idx = i;
        }
        
        if (x_idx < 0 || y_idx < 0 || z_idx < 0) {
            ROS_ERROR("Point cloud missing xyz fields");
            return false;
        }
        
        // Parse point cloud
        points.reserve(msg->width * msg->height);
        
        // Get transform as Eigen
        Eigen::Affine3d tf_eigen = tf2::transformToEigen(transform);
        
        for (size_t i = 0; i < msg->width * msg->height; ++i) {
            const uint8_t* ptr = &msg->data[i * msg->point_step];
            
            // Extract position
            float x, y, z;
            memcpy(&x, ptr + msg->fields[x_idx].offset, sizeof(float));
            memcpy(&y, ptr + msg->fields[y_idx].offset, sizeof(float));
            memcpy(&z, ptr + msg->fields[z_idx].offset, sizeof(float));
            
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }
            
            // Transform to map frame
            Eigen::Vector3d pt_sensor(x, y, z);
            Eigen::Vector3d pt_map = tf_eigen * pt_sensor;
            
            hesfm::SemanticPoint sp;
            sp.position = pt_map;
            sp.depth = std::sqrt(x*x + y*y + z*z);
            
            // Extract label
            if (label_idx >= 0) {
                uint32_t label;
                memcpy(&label, ptr + msg->fields[label_idx].offset, sizeof(uint32_t));
                sp.semantic_class = static_cast<int>(label);
            }
            
            // Extract uncertainty
            if (uncertainty_idx >= 0) {
                float unc;
                memcpy(&unc, ptr + msg->fields[uncertainty_idx].offset, sizeof(float));
                sp.uncertainty_semantic = unc;
                sp.uncertainty_total = unc;
            } else {
                sp.uncertainty_semantic = 0.3;
                sp.uncertainty_total = 0.3;
            }
            
            // Extract RGB
            if (rgb_idx >= 0) {
                uint32_t rgb;
                memcpy(&rgb, ptr + msg->fields[rgb_idx].offset, sizeof(uint32_t));
                sp.r = (rgb >> 16) & 0xFF;
                sp.g = (rgb >> 8) & 0xFF;
                sp.b = rgb & 0xFF;
            }
            
            // Initialize class probabilities (uniform if not provided)
            sp.class_probabilities.resize(config_.map.num_classes, 
                                          1.0 / config_.map.num_classes);
            if (sp.semantic_class >= 0 && sp.semantic_class < config_.map.num_classes) {
                // Set high probability for predicted class
                std::fill(sp.class_probabilities.begin(), sp.class_probabilities.end(), 0.01);
                sp.class_probabilities[sp.semantic_class] = 0.9;
            }
            
            // Check if traversable
            sp.is_traversable = (std::find(traversable_classes_.begin(),
                                           traversable_classes_.end(),
                                           sp.semantic_class) != traversable_classes_.end());
            
            points.push_back(sp);
        }
        
        return true;
    }
    
    // =========================================================================
    // Publishing
    // =========================================================================
    
    void publishSemanticMap() {
        if (map_pub_.getNumSubscribers() == 0) return;
        
        auto cells = pipeline_->getMap().getOccupiedCells();
        if (cells.empty()) return;
        
        // Create point cloud
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.reserve(cells.size());
        
        for (const auto& cell : cells) {
            pcl::PointXYZRGB pt;
            pt.x = cell.position.x();
            pt.y = cell.position.y();
            pt.z = cell.position.z();
            
            int pred_class = cell.state.getPredictedClass();

            // Color by semantic class
            if (pred_class >= 0 && pred_class < static_cast<int>(class_colors_.size())) {
                pt.r = class_colors_[pred_class][0];
                pt.g = class_colors_[pred_class][1];
                pt.b = class_colors_[pred_class][2];
            } else {
                pt.r = pt.g = pt.b = 128;
            }
            
            cloud.push_back(pt);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        
        map_pub_.publish(msg);
    }
    
    void publishCostmap() {
        if (costmap_pub_.getNumSubscribers() == 0) return;
        
        int width, height;
        auto costmap_data = pipeline_->getCostmap(width, height);
        
        if (costmap_data.empty()) return;
        
        nav_msgs::OccupancyGrid msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        
        msg.info.resolution = config_.map.resolution;
        msg.info.width = width;
        msg.info.height = height;
        msg.info.origin.position.x = config_.map.origin_x;
        msg.info.origin.position.y = config_.map.origin_y;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;
        
        msg.data = costmap_data;
        
        costmap_pub_.publish(msg);
    }
    
    void publishPrimitives() {
        if (primitives_pub_.getNumSubscribers() == 0) return;
        
        visualization_msgs::MarkerArray markers;
        
        // Delete all previous markers
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        markers.markers.push_back(delete_marker);
        
        for (size_t i = 0; i < current_primitives_.size(); ++i) {
            const auto& prim = current_primitives_[i];
            
            visualization_msgs::Marker marker;
            marker.header.frame_id = map_frame_;
            marker.header.stamp = ros::Time::now();
            marker.ns = "primitives";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            
            // Position
            marker.pose.position.x = prim.centroid.x();
            marker.pose.position.y = prim.centroid.y();
            marker.pose.position.z = prim.centroid.z();
            
            // Orientation from covariance eigenvectors
            auto orientation = prim.getOrientation();
            marker.pose.orientation.x = orientation.x();
            marker.pose.orientation.y = orientation.y();
            marker.pose.orientation.z = orientation.z();
            marker.pose.orientation.w = orientation.w();
            
            // Scale from eigenvalues
            auto eigenvalues = prim.getEigenvalues();
            marker.scale.x = 2.0 * std::sqrt(std::max(0.01, eigenvalues(0)));
            marker.scale.y = 2.0 * std::sqrt(std::max(0.01, eigenvalues(1)));
            marker.scale.z = 2.0 * std::sqrt(std::max(0.01, eigenvalues(2)));
            
            // Color by uncertainty (green = low, red = high)
            marker.color.r = prim.uncertainty;
            marker.color.g = 1.0 - prim.uncertainty;
            marker.color.b = 0.0;
            marker.color.a = 0.5;
            
            marker.lifetime = ros::Duration(0.5);
            
            markers.markers.push_back(marker);
        }
        
        primitives_pub_.publish(markers);
    }
    
    void publishUncertaintyMap() {
        if (uncertainty_map_pub_.getNumSubscribers() == 0) return;
        
        auto cells = pipeline_->getMap().getOccupiedCells();
        if (cells.empty()) return;
        
        pcl::PointCloud<pcl::PointXYZI> cloud;
        cloud.reserve(cells.size());
        
        for (const auto& cell : cells) {
            pcl::PointXYZI pt;
            pt.x = cell.position.x();
            pt.y = cell.position.y();
            pt.z = cell.position.z();
            pt.intensity = 1.0 - cell.state.getConfidence();
            
            cloud.push_back(pt);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        
        uncertainty_map_pub_.publish(msg);
    }
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    void updateStatistics(double processing_time, size_t num_points, int num_primitives) {
        total_frames_++;
        total_points_ += num_points;
        total_processing_time_ += processing_time;
        
        if (total_frames_ % 100 == 0) {
            double avg_time = total_processing_time_ / total_frames_;
            double fps = 1000.0 / avg_time;
            ROS_INFO("HESFM Stats: %.1f FPS, %zu cells, %.1f%% coverage",
                     fps, pipeline_->getMap().getNumCells(),
                     pipeline_->getMap().getCoverage() * 100.0);
        }
    }
    
    // =========================================================================
    // Member Variables
    // =========================================================================
    
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Publishers
    ros::Publisher map_pub_;
    ros::Publisher costmap_pub_;
    ros::Publisher primitives_pub_;
    ros::Publisher uncertainty_map_pub_;
    
    // Subscribers
    ros::Subscriber cloud_sub_;
    
    // Services
    ros::ServiceServer reset_srv_;
    
    // Timers
    ros::Timer map_timer_;
    ros::Timer costmap_timer_;
    
    // HESFM
    hesfm::HESFMConfig config_;
    std::unique_ptr<hesfm::HESFMPipeline> pipeline_;
    std::vector<hesfm::GaussianPrimitive> current_primitives_;
    
    // Parameters
    std::string map_frame_;
    std::string sensor_frame_;
    std::vector<int> traversable_classes_;
    std::vector<std::array<uint8_t, 3>> class_colors_;
    
    // Statistics
    int total_frames_ = 0;
    size_t total_points_ = 0;
    double total_processing_time_ = 0.0;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "hesfm_mapper_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    try {
        HESFMMapperNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    
    return 0;
}
