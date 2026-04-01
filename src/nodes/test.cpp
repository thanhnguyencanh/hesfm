/**
 * @file hesfm_mapper_node.cpp
 * @brief Main ROS node for HESFM semantic mapping with 3D Costmap
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node subscribes to semantic point clouds and maintains
 * the semantic map using the HESFM framework.
 * 
 * NEW: Integrated 3D Semantic Costmap (SLIDE SLAM inspired)
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

// =============================================================================
// 3D Semantic Costmap (SLIDE SLAM Inspired)
// =============================================================================

/**
 * @brief Semantic traversability costs for indoor classes
 */
class SemanticTraversabilityCosts {
public:
    /**
     * @brief Get default costs for SUN RGB-D or NYUv2
     */
    static std::vector<float> getDefaultCosts(const std::string& dataset, int num_classes) {
        std::vector<float> costs(num_classes, 0.5f);  // Default: uncertain
        
        if (dataset == "sunrgbd" && num_classes >= 37) {
            // SUN RGB-D 37 classes
            costs[0] = 1.0f;   // wall - lethal
            costs[1] = 0.0f;   // floor - traversable
            costs[2] = 0.85f;  // cabinet
            costs[3] = 0.8f;   // bed
            costs[4] = 0.6f;   // chair
            costs[5] = 0.8f;   // sofa
            costs[6] = 0.7f;   // table
            costs[7] = 0.95f;  // door
            costs[8] = 1.0f;   // window
            costs[9] = 0.85f;  // bookshelf
            costs[10] = 0.0f;  // picture (on wall)
            costs[11] = 0.7f;  // counter
            costs[12] = 0.2f;  // blinds
            costs[13] = 0.7f;  // desk
            costs[14] = 0.85f; // shelves
            costs[15] = 0.3f;  // curtain
            costs[16] = 0.85f; // dresser
            costs[17] = 0.5f;  // pillow
            costs[18] = 1.0f;  // mirror
            costs[19] = 0.1f;  // floor_mat - traversable
            costs[20] = 0.4f;  // clothes
            costs[21] = 0.0f;  // ceiling - ignore (overhead)
            costs[22] = 0.5f;  // books
            costs[23] = 0.9f;  // fridge
            costs[24] = 0.8f;  // television
            costs[25] = 0.3f;  // paper
            costs[26] = 0.3f;  // towel
            costs[27] = 0.3f;  // shower_curtain
            costs[28] = 0.5f;  // box
            costs[29] = 0.95f; // whiteboard
            costs[30] = 0.9f;  // person (dynamic)
            costs[31] = 0.8f;  // night_stand
            costs[32] = 0.95f; // toilet
            costs[33] = 0.95f; // sink
            costs[34] = 0.6f;  // lamp
            costs[35] = 0.95f; // bathtub
            costs[36] = 0.4f;  // bag
        } else if (num_classes >= 40) {
            // NYUv2 40 classes - similar mapping
            costs[0] = 1.0f;   // wall
            costs[1] = 0.0f;   // floor
            costs[2] = 0.85f;  // cabinet
            costs[3] = 0.8f;   // bed
            costs[4] = 0.6f;   // chair
            costs[5] = 0.8f;   // sofa
            costs[6] = 0.7f;   // table
            costs[7] = 0.95f;  // door
            costs[8] = 1.0f;   // window
            costs[9] = 0.85f;  // bookshelf
            costs[10] = 0.0f;  // picture
            costs[11] = 0.7f;  // counter
            costs[12] = 0.2f;  // blinds
            costs[13] = 0.7f;  // desk
            costs[14] = 0.85f; // shelves
            costs[15] = 0.3f;  // curtain
            costs[16] = 0.85f; // dresser
            costs[17] = 0.5f;  // pillow
            costs[18] = 1.0f;  // mirror
            costs[19] = 0.1f;  // floor_mat
            costs[20] = 0.4f;  // clothes
            costs[21] = 0.0f;  // ceiling
            costs[22] = 0.5f;  // books
            costs[23] = 0.9f;  // fridge
            costs[24] = 0.8f;  // television
            costs[25] = 0.3f;  // paper
            costs[26] = 0.3f;  // towel
            costs[27] = 0.3f;  // shower_curtain
            costs[28] = 0.5f;  // box
            costs[29] = 0.95f; // whiteboard
            costs[30] = 0.9f;  // person
            costs[31] = 0.8f;  // night_stand
            costs[32] = 0.95f; // toilet
            costs[33] = 0.95f; // sink
            costs[34] = 0.6f;  // lamp
            costs[35] = 0.95f; // bathtub
            costs[36] = 0.4f;  // bag
            costs[37] = 0.6f;  // otherstructure
            costs[38] = 0.6f;  // otherfurniture
            costs[39] = 0.5f;  // otherprop
        }
        
        return costs;
    }
};

/**
 * @brief 3D Semantic Costmap configuration
 */
struct Costmap3DConfig {
    bool enabled = true;
    double voxel_size = 0.1;
    double map_size_x = 15.0;
    double map_size_y = 15.0;
    double map_size_z = 2.5;
    double robot_height = 0.5;
    double robot_radius = 0.3;
    double ground_clearance = 0.05;
    double uncertainty_weight = 0.2;
    double lethal_threshold = 0.9;
    double publish_rate = 3.0;
    std::string dataset = "sunrgbd";
};

/**
 * @brief HESFM Mapper ROS Node with integrated 3D Costmap
 */
class HESFMMapperNode {
public:
    HESFMMapperNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), tf_listener_(tf_buffer_) {
        
        // Load parameters
        loadParameters();
        
        // Initialize HESFM components
        initializeHESFM();
        
        // Initialize 3D Costmap
        initializeCostmap3D();
        
        // Setup ROS interfaces
        setupSubscribers();
        setupPublishers();
        setupServices();
        setupTimers();
        
        ROS_INFO("HESFM Mapper Node initialized");
        ROS_INFO("  Map frame: %s", map_frame_.c_str());
        ROS_INFO("  Resolution: %.3f m", config_.map.resolution);
        ROS_INFO("  Num classes: %d", config_.map.num_classes);
        ROS_INFO("  3D Costmap: %s", costmap_3d_config_.enabled ? "ENABLED" : "DISABLED");
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
        pnh_.param<std::string>("robot_frame", robot_frame_, "base_link");
        
        // Dataset
        pnh_.param<std::string>("dataset", dataset_, "sunrgbd");
        
        // Map parameters
        pnh_.param("resolution", config_.map.resolution, 0.05);
        pnh_.param("num_classes", config_.map.num_classes, 40);
        pnh_.param("map_size_x", config_.map.size_x, 20.0);
        pnh_.param("map_size_y", config_.map.size_y, 20.0);
        pnh_.param("map_size_z", config_.map.size_z, 3.0);
        pnh_.param("origin_x", config_.map.origin_x, -10.0);
        pnh_.param("origin_y", config_.map.origin_y, -10.0);
        pnh_.param("origin_z", config_.map.origin_z, -0.5);
        
        // Uncertainty weights
        pnh_.param("w_semantic", config_.uncertainty.w_semantic, 0.4);
        pnh_.param("w_spatial", config_.uncertainty.w_spatial, 0.2);
        pnh_.param("w_observation", config_.uncertainty.w_observation, 0.25);
        pnh_.param("w_temporal", config_.uncertainty.w_temporal, 0.15);
        
        // Gaussian primitive parameters
        pnh_.param("target_primitives", config_.primitive.target_primitives, 128);
        pnh_.param("min_points_per_primitive", config_.primitive.min_points_per_primitive, 5);
        pnh_.param("conflict_threshold", config_.primitive.conflict_threshold, 0.3);
        
        // Kernel parameters
        pnh_.param("length_scale_min", config_.kernel.length_scale_min, 0.1);
        pnh_.param("length_scale_max", config_.kernel.length_scale_max, 0.5);
        pnh_.param("uncertainty_threshold", config_.kernel.uncertainty_threshold, 0.7);
        pnh_.param("gamma", config_.kernel.gamma, 2.0);
        
        // Navigation parameters
        pnh_.param("costmap_height_min", config_.navigation.costmap_height_min, 0.0);
        pnh_.param("costmap_height_max", config_.navigation.costmap_height_max, 0.5);
        
        // Processing parameters
        pnh_.param("map_publish_rate", config_.processing.map_publish_rate, 2.0);
        pnh_.param("costmap_publish_rate", config_.processing.costmap_publish_rate, 5.0);
        
        // Traversable classes
        std::vector<int> traversable_default = {1, 19};  // floor, floor_mat
        pnh_.param("traversable_classes", traversable_classes_, traversable_default);
        
        // =====================================================================
        // 3D Costmap Parameters (SLIDE SLAM inspired)
        // =====================================================================
        pnh_.param("enable_costmap_3d", costmap_3d_config_.enabled, true);
        pnh_.param("costmap_3d/voxel_size", costmap_3d_config_.voxel_size, 0.1);
        pnh_.param("costmap_3d/map_size_x", costmap_3d_config_.map_size_x, 15.0);
        pnh_.param("costmap_3d/map_size_y", costmap_3d_config_.map_size_y, 15.0);
        pnh_.param("costmap_3d/map_size_z", costmap_3d_config_.map_size_z, 2.5);
        pnh_.param("costmap_3d/robot_height", costmap_3d_config_.robot_height, 0.5);
        pnh_.param("costmap_3d/robot_radius", costmap_3d_config_.robot_radius, 0.3);
        pnh_.param("costmap_3d/ground_clearance", costmap_3d_config_.ground_clearance, 0.05);
        pnh_.param("costmap_3d/uncertainty_weight", costmap_3d_config_.uncertainty_weight, 0.2);
        pnh_.param("costmap_3d/lethal_threshold", costmap_3d_config_.lethal_threshold, 0.9);
        pnh_.param("costmap_3d/publish_rate", costmap_3d_config_.publish_rate, 3.0);
        costmap_3d_config_.dataset = dataset_;
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
    
    void initializeCostmap3D() {
        if (!costmap_3d_config_.enabled) {
            ROS_INFO("3D Costmap disabled");
            return;
        }
        
        // Initialize traversability costs based on dataset
        class_costs_ = SemanticTraversabilityCosts::getDefaultCosts(
            costmap_3d_config_.dataset, config_.map.num_classes);
        
        ROS_INFO("3D Costmap initialized with %s traversability costs",
                 costmap_3d_config_.dataset.c_str());
    }
    
    void setupSubscribers() {
        // Semantic point cloud subscriber
        cloud_sub_ = nh_.subscribe("semantic_cloud", 1, 
                                    &HESFMMapperNode::cloudCallback, this);
        
        // Odometry subscriber for robot pose (for 3D costmap)
        if (costmap_3d_config_.enabled) {
            odom_sub_ = nh_.subscribe("odom", 1, 
                                       &HESFMMapperNode::odomCallback, this);
        }
        
        ROS_INFO("Subscribed to: %s", cloud_sub_.getTopic().c_str());
    }
    
    void setupPublishers() {
        // Semantic map publisher
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_map", 1);
        
        // Costmap publisher (original 2D)
        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("costmap", 1);
        
        // Primitives visualization publisher
        primitives_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("primitives", 1);
        
        // Uncertainty map publisher
        uncertainty_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("uncertainty_map", 1);
        
        // =====================================================================
        // 3D Costmap Publishers (SLIDE SLAM inspired)
        // =====================================================================
        if (costmap_3d_config_.enabled) {
            // 3D costmap visualization
            costmap_3d_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
                "semantic_costmap_3d", 1);
            
            // 2D semantic costmap projection for move_base
            costmap_2d_semantic_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(
                "semantic_costmap_2d", 1);
            
            // Traversability point cloud
            traversability_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
                "traversability_cloud", 1);
            
            ROS_INFO("3D Costmap publishers initialized");
        }
        
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
        
        // 3D Costmap publishing timer
        if (costmap_3d_config_.enabled) {
            costmap_3d_timer_ = nh_.createTimer(
                ros::Duration(1.0 / costmap_3d_config_.publish_rate),
                &HESFMMapperNode::costmap3DPublishCallback, this);
        }
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
    
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        robot_pose_ = msg->pose.pose;
        has_robot_pose_ = true;
    }
    
    void mapPublishCallback(const ros::TimerEvent&) {
        publishSemanticMap();
        publishPrimitives();
        publishUncertaintyMap();
    }
    
    void costmapPublishCallback(const ros::TimerEvent&) {
        publishCostmap();
    }
    
    void costmap3DPublishCallback(const ros::TimerEvent&) {
        if (!costmap_3d_config_.enabled) return;
        
        publishCostmap3D();
        publishCostmap2DSemantic();
        publishTraversabilityCloud();
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
    // Publishing - Original
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
            
            // Color by class
            uint8_t r = static_cast<uint8_t>((pred_class * 37) % 256);
            uint8_t g = static_cast<uint8_t>((pred_class * 91) % 256);
            uint8_t b = static_cast<uint8_t>((pred_class * 157) % 256);
            
            pt.r = r;
            pt.g = g;
            pt.b = b;
            
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
    // Publishing - 3D Semantic Costmap (SLIDE SLAM Inspired)
    // =========================================================================
    
    /**
     * @brief Compute traversability cost for a voxel
     */
    float computeTraversabilityCost(int semantic_class, float uncertainty, double height) {
        // Get base semantic cost
        float cost = (semantic_class >= 0 && 
                      semantic_class < static_cast<int>(class_costs_.size()))
                    ? class_costs_[semantic_class]
                    : 0.5f;
        
        // Add uncertainty penalty
        cost += costmap_3d_config_.uncertainty_weight * uncertainty;
        
        // Height-based filtering
        if (height > costmap_3d_config_.robot_height) {
            // Above robot - not an obstacle (ceiling, hanging objects)
            cost *= 0.1f;
        } else if (height < costmap_3d_config_.ground_clearance) {
            // Below ground clearance - check if traversable
            if (std::find(traversable_classes_.begin(), traversable_classes_.end(),
                         semantic_class) != traversable_classes_.end()) {
                cost = 0.0f;
            }
        }
        
        return std::min(1.0f, std::max(0.0f, cost));
    }
    
    void publishCostmap3D() {
        if (costmap_3d_pub_.getNumSubscribers() == 0) return;
        
        auto cells = pipeline_->getMap().getOccupiedCells();
        if (cells.empty()) return;
        
        // Get robot position
        double robot_x = 0.0, robot_y = 0.0, robot_z = 0.0;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            if (has_robot_pose_) {
                robot_x = robot_pose_.position.x;
                robot_y = robot_pose_.position.y;
                robot_z = robot_pose_.position.z;
            }
        }
        
        visualization_msgs::MarkerArray markers;
        visualization_msgs::Marker marker;
        
        marker.header.frame_id = map_frame_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "semantic_costmap_3d";
        marker.type = visualization_msgs::Marker::CUBE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.id = 0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = costmap_3d_config_.voxel_size * 0.9;
        marker.scale.y = costmap_3d_config_.voxel_size * 0.9;
        marker.scale.z = costmap_3d_config_.voxel_size * 0.9;
        
        double max_dist_sq = (costmap_3d_config_.map_size_x / 2) * 
                             (costmap_3d_config_.map_size_x / 2);
        
        for (const auto& cell : cells) {
            double x = cell.position.x();
            double y = cell.position.y();
            double z = cell.position.z();
            
            // Check if within local range
            double dx = x - robot_x;
            double dy = y - robot_y;
            if (dx*dx + dy*dy > max_dist_sq) continue;
            
            // Get semantic info
            int pred_class = cell.state.getPredictedClass();
            float uncertainty = 1.0f - cell.state.getConfidence();
            
            // Compute cost
            float cost = computeTraversabilityCost(pred_class, uncertainty, z - robot_z);
            
            // Skip low-cost voxels for clarity
            if (cost < 0.3f) continue;
            
            // Add point
            geometry_msgs::Point pt;
            pt.x = x; pt.y = y; pt.z = z;
            marker.points.push_back(pt);
            
            // Color by cost: green -> yellow -> red
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
        
        markers.markers.push_back(marker);
        costmap_3d_pub_.publish(markers);
    }
    
    void publishCostmap2DSemantic() {
        if (costmap_2d_semantic_pub_.getNumSubscribers() == 0) return;
        
        auto cells = pipeline_->getMap().getOccupiedCells();
        if (cells.empty()) return;
        
        // Get robot position
        double robot_x = 0.0, robot_y = 0.0, robot_z = 0.0;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            if (has_robot_pose_) {
                robot_x = robot_pose_.position.x;
                robot_y = robot_pose_.position.y;
                robot_z = robot_pose_.position.z;
            }
        }
        
        // Grid setup
        int grid_size = static_cast<int>(costmap_3d_config_.map_size_x / 
                                         costmap_3d_config_.voxel_size);
        
        nav_msgs::OccupancyGrid grid;
        grid.header.frame_id = map_frame_;
        grid.header.stamp = ros::Time::now();
        grid.info.resolution = costmap_3d_config_.voxel_size;
        grid.info.width = grid_size;
        grid.info.height = grid_size;
        grid.info.origin.position.x = robot_x - costmap_3d_config_.map_size_x / 2;
        grid.info.origin.position.y = robot_y - costmap_3d_config_.map_size_y / 2;
        grid.info.origin.position.z = 0;
        grid.info.origin.orientation.w = 1.0;
        
        grid.data.assign(grid_size * grid_size, -1);  // Unknown
        
        // Track max cost per cell
        std::vector<float> cell_max_cost(grid_size * grid_size, -1.0f);
        
        for (const auto& cell : cells) {
            double x = cell.position.x();
            double y = cell.position.y();
            double z = cell.position.z();
            
            // Height filtering for robot body
            double height = z - robot_z;
            if (height < costmap_3d_config_.ground_clearance || 
                height > costmap_3d_config_.robot_height) {
                continue;
            }
            
            // Convert to grid cell
            int gx = static_cast<int>((x - grid.info.origin.position.x) / 
                                      costmap_3d_config_.voxel_size);
            int gy = static_cast<int>((y - grid.info.origin.position.y) / 
                                      costmap_3d_config_.voxel_size);
            
            if (gx < 0 || gx >= grid_size || gy < 0 || gy >= grid_size) continue;
            
            int cell_idx = gy * grid_size + gx;
            
            // Get semantics and compute cost
            int pred_class = cell.state.getPredictedClass();
            float uncertainty = 1.0f - cell.state.getConfidence();
            float cost = computeTraversabilityCost(pred_class, uncertainty, height);
            
            // Track maximum
            if (cost > cell_max_cost[cell_idx]) {
                cell_max_cost[cell_idx] = cost;
            }
        }
        
        // Convert to occupancy values
        for (size_t i = 0; i < cell_max_cost.size(); ++i) {
            if (cell_max_cost[i] >= 0.0f) {
                grid.data[i] = static_cast<int8_t>(cell_max_cost[i] * 100);
            }
        }
        
        costmap_2d_semantic_pub_.publish(grid);
    }
    
    void publishTraversabilityCloud() {
        if (traversability_cloud_pub_.getNumSubscribers() == 0) return;
        
        auto cells = pipeline_->getMap().getOccupiedCells();
        if (cells.empty()) return;
        
        // Get robot position
        double robot_z = 0.0;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            if (has_robot_pose_) {
                robot_z = robot_pose_.position.z;
            }
        }
        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.reserve(cells.size());
        
        for (const auto& cell : cells) {
            int pred_class = cell.state.getPredictedClass();
            float uncertainty = 1.0f - cell.state.getConfidence();
            float cost = computeTraversabilityCost(pred_class, uncertainty, 
                                                   cell.position.z() - robot_z);
            
            pcl::PointXYZRGB pt;
            pt.x = cell.position.x();
            pt.y = cell.position.y();
            pt.z = cell.position.z();
            
            // Color by traversability
            if (cost < 0.3f) {
                // Green - traversable
                pt.r = 0; pt.g = 255; pt.b = 0;
            } else if (cost < 0.6f) {
                // Yellow - caution
                pt.r = 255; pt.g = 255; pt.b = 0;
            } else if (cost < 0.9f) {
                // Orange - high cost
                pt.r = 255; pt.g = 128; pt.b = 0;
            } else {
                // Red - lethal
                pt.r = 255; pt.g = 0; pt.b = 0;
            }
            
            cloud.push_back(pt);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        
        traversability_cloud_pub_.publish(msg);
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
    
    // Publishers - Original
    ros::Publisher map_pub_;
    ros::Publisher costmap_pub_;
    ros::Publisher primitives_pub_;
    ros::Publisher uncertainty_map_pub_;
    
    // Publishers - 3D Costmap (NEW)
    ros::Publisher costmap_3d_pub_;
    ros::Publisher costmap_2d_semantic_pub_;
    ros::Publisher traversability_cloud_pub_;
    
    // Subscribers
    ros::Subscriber cloud_sub_;
    ros::Subscriber odom_sub_;
    
    // Services
    ros::ServiceServer reset_srv_;
    
    // Timers
    ros::Timer map_timer_;
    ros::Timer costmap_timer_;
    ros::Timer costmap_3d_timer_;
    
    // HESFM
    hesfm::HESFMConfig config_;
    std::unique_ptr<hesfm::HESFMPipeline> pipeline_;
    std::vector<hesfm::GaussianPrimitive> current_primitives_;
    
    // 3D Costmap (SLIDE SLAM inspired)
    Costmap3DConfig costmap_3d_config_;
    std::vector<float> class_costs_;
    
    // Robot state
    geometry_msgs::Pose robot_pose_;
    std::mutex odom_mutex_;
    bool has_robot_pose_ = false;
    
    // Parameters
    std::string map_frame_;
    std::string sensor_frame_;
    std::string robot_frame_;
    std::string dataset_;
    std::vector<int> traversable_classes_;
    
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