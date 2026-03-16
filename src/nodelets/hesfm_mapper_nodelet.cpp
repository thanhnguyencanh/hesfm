/**
 * @file hesfm_mapper_nodelet.cpp
 * @brief HESFM Mapper Nodelet for zero-copy data transfer
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This nodelet provides the same functionality as hesfm_mapper_node
 * but with zero-copy intraprocess communication when used with
 * other nodelets in the same nodelet manager.
 * 
 * Benefits:
 * - Zero-copy for PointCloud2 messages
 * - Reduced latency
 * - Lower CPU overhead
 * 
 * Usage:
 *   rosrun nodelet nodelet manager __name:=hesfm_manager
 *   rosrun nodelet nodelet load hesfm/HESFMMapperNodelet hesfm_manager
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <boost/thread.hpp>
#include <memory>
#include <queue>

#include "hesfm/hesfm.h"

namespace hesfm {

/**
 * @brief HESFM Mapper Nodelet
 */
class HESFMMapperNodelet : public nodelet::Nodelet {
public:
    HESFMMapperNodelet() = default;
    virtual ~HESFMMapperNodelet() {
        if (processing_thread_.joinable()) {
            running_ = false;
            processing_thread_.join();
        }
    }

private:
    virtual void onInit() override {
        nh_ = getNodeHandle();
        pnh_ = getPrivateNodeHandle();
        
        // Initialize TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Load parameters
        loadParameters();
        
        // Initialize HESFM
        initializeHESFM();
        
        // Setup ROS interfaces
        setupSubscribers();
        setupPublishers();
        setupServices();
        setupTimers();
        
        // Start processing thread (optional async mode)
        if (async_processing_) {
            running_ = true;
            processing_thread_ = boost::thread(&HESFMMapperNodelet::processingLoop, this);
        }
        
        NODELET_INFO("HESFM Mapper Nodelet initialized");
        NODELET_INFO("  Map frame: %s", map_frame_.c_str());
        NODELET_INFO("  Resolution: %.3f m", config_.map.resolution);
    }
    
    void loadParameters() {
        pnh_.param<std::string>("map_frame", map_frame_, "map");
        pnh_.param<std::string>("sensor_frame", sensor_frame_, "camera_color_optical_frame");
        
        pnh_.param("resolution", config_.map.resolution, 0.05);
        pnh_.param("num_classes", config_.map.num_classes, 40);
        pnh_.param("map_size_x", config_.map.size_x, 20.0);
        pnh_.param("map_size_y", config_.map.size_y, 20.0);
        pnh_.param("map_size_z", config_.map.size_z, 3.0);
        
        pnh_.param("w_semantic", config_.uncertainty.w_semantic, 0.4);
        pnh_.param("w_spatial", config_.uncertainty.w_spatial, 0.2);
        pnh_.param("w_observation", config_.uncertainty.w_observation, 0.25);
        pnh_.param("w_temporal", config_.uncertainty.w_temporal, 0.15);
        
        pnh_.param("target_primitives", config_.primitive.target_primitives, 128);
        pnh_.param("length_scale_min", config_.kernel.length_scale_min, 0.1);
        pnh_.param("length_scale_max", config_.kernel.length_scale_max, 0.5);
        
        pnh_.param("map_publish_rate", config_.processing.map_publish_rate, 2.0);
        pnh_.param("costmap_publish_rate", config_.processing.costmap_publish_rate, 5.0);
        
        pnh_.param("async_processing", async_processing_, false);
        
        std::vector<int> trav_default = {1, 19};
        pnh_.param("traversable_classes", traversable_classes_, trav_default);
    }
    
    void initializeHESFM() {
        pipeline_ = std::make_unique<HESFMPipeline>(config_);
        
        std::set<int> trav_set(traversable_classes_.begin(), traversable_classes_.end());
        pipeline_->getMap().setTraversableClasses(trav_set);
        pipeline_->getKernel().setTraversableClasses(trav_set);
    }
    
    void setupSubscribers() {
        // Use getMTNodeHandle() for thread-safe callbacks
        cloud_sub_ = nh_.subscribe("semantic_cloud", 1, 
                                    &HESFMMapperNodelet::cloudCallback, this);
    }
    
    void setupPublishers() {
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_map", 1);
        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("costmap", 1);
        primitives_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("primitives", 1);
        uncertainty_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("uncertainty_map", 1);
    }
    
    void setupServices() {
        reset_srv_ = nh_.advertiseService("reset_map", 
                                           &HESFMMapperNodelet::resetMapCallback, this);
    }
    
    void setupTimers() {
        map_timer_ = nh_.createTimer(
            ros::Duration(1.0 / config_.processing.map_publish_rate),
            &HESFMMapperNodelet::mapPublishCallback, this);
        
        costmap_timer_ = nh_.createTimer(
            ros::Duration(1.0 / config_.processing.costmap_publish_rate),
            &HESFMMapperNodelet::costmapPublishCallback, this);
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (async_processing_) {
            // Queue for async processing
            boost::mutex::scoped_lock lock(queue_mutex_);
            cloud_queue_.push(msg);
            queue_condition_.notify_one();
        } else {
            // Process synchronously
            processCloud(msg);
        }
    }
    
    void processCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        // Get transform
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(
                map_frame_, msg->header.frame_id,
                msg->header.stamp, ros::Duration(0.1));
        } catch (tf2::TransformException& ex) {
            NODELET_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return;
        }
        
        Vector3d sensor_origin(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z);
        
        // Convert point cloud
        std::vector<SemanticPoint> points;
        convertPointCloud(msg, transform, points);
        
        if (points.empty()) return;
        
        // Process through pipeline
        boost::mutex::scoped_lock lock(pipeline_mutex_);
        pipeline_->process(points, sensor_origin);
        current_primitives_ = pipeline_->getPrimitiveBuilder().buildPrimitives(points);
    }
    
    void processingLoop() {
        while (running_) {
            sensor_msgs::PointCloud2::ConstPtr msg;
            
            {
                boost::mutex::scoped_lock lock(queue_mutex_);
                while (cloud_queue_.empty() && running_) {
                    queue_condition_.timed_wait(lock, boost::posix_time::milliseconds(100));
                }
                
                if (!running_) break;
                
                msg = cloud_queue_.front();
                cloud_queue_.pop();
            }
            
            processCloud(msg);
        }
    }
    
    bool convertPointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg,
                           const geometry_msgs::TransformStamped& transform,
                           std::vector<SemanticPoint>& points) {
        
        int x_idx = -1, y_idx = -1, z_idx = -1, label_idx = -1, unc_idx = -1;
        
        for (size_t i = 0; i < msg->fields.size(); ++i) {
            const auto& field = msg->fields[i];
            if (field.name == "x") x_idx = i;
            else if (field.name == "y") y_idx = i;
            else if (field.name == "z") z_idx = i;
            else if (field.name == "label") label_idx = i;
            else if (field.name == "uncertainty") unc_idx = i;
        }
        
        if (x_idx < 0 || y_idx < 0 || z_idx < 0) return false;
        
        Eigen::Affine3d tf_eigen = tf2::transformToEigen(transform);
        points.reserve(msg->width * msg->height);
        
        for (size_t i = 0; i < msg->width * msg->height; ++i) {
            const uint8_t* ptr = &msg->data[i * msg->point_step];
            
            float x, y, z;
            memcpy(&x, ptr + msg->fields[x_idx].offset, sizeof(float));
            memcpy(&y, ptr + msg->fields[y_idx].offset, sizeof(float));
            memcpy(&z, ptr + msg->fields[z_idx].offset, sizeof(float));
            
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            
            Eigen::Vector3d pt_map = tf_eigen * Eigen::Vector3d(x, y, z);
            
            SemanticPoint sp;
            sp.position = pt_map;
            sp.depth = std::sqrt(x*x + y*y + z*z);
            
            if (label_idx >= 0) {
                uint32_t label;
                memcpy(&label, ptr + msg->fields[label_idx].offset, sizeof(uint32_t));
                sp.semantic_class = static_cast<int>(label);
            }
            
            if (unc_idx >= 0) {
                float unc;
                memcpy(&unc, ptr + msg->fields[unc_idx].offset, sizeof(float));
                sp.uncertainty_total = unc;
            } else {
                sp.uncertainty_total = 0.3;
            }
            
            sp.class_probabilities.resize(config_.map.num_classes, 1.0 / config_.map.num_classes);
            if (sp.semantic_class >= 0 && sp.semantic_class < config_.map.num_classes) {
                std::fill(sp.class_probabilities.begin(), sp.class_probabilities.end(), 0.01);
                sp.class_probabilities[sp.semantic_class] = 0.9;
            }
            
            points.push_back(sp);
        }
        
        return true;
    }
    
    void mapPublishCallback(const ros::TimerEvent&) {
        publishSemanticMap();
        publishPrimitives();
    }
    
    void costmapPublishCallback(const ros::TimerEvent&) {
        publishCostmap();
    }
    
    bool resetMapCallback(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {
        boost::mutex::scoped_lock lock(pipeline_mutex_);
        pipeline_->reset();
        NODELET_INFO("Map reset");
        return true;
    }
    
    void publishSemanticMap() {
        if (map_pub_.getNumSubscribers() == 0) return;
        
        boost::mutex::scoped_lock lock(pipeline_mutex_);
        auto cells = pipeline_->getMap().getOccupiedCells();
        if (cells.empty()) return;
        
        sensor_msgs::PointCloud2 msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        msg.height = 1;
        msg.width = cells.size();
        
        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2Fields(4,
            "x", 1, sensor_msgs::PointField::FLOAT32,
            "y", 1, sensor_msgs::PointField::FLOAT32,
            "z", 1, sensor_msgs::PointField::FLOAT32,
            "rgb", 1, sensor_msgs::PointField::UINT32);
        modifier.resize(cells.size());
        
        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgb(msg, "rgb");
        
        for (const auto& cell : cells) {
            *iter_x = cell.position.x();
            *iter_y = cell.position.y();
            *iter_z = cell.position.z();
            
            int cls = cell.state.getPredictedClass();
            uint8_t r = static_cast<uint8_t>((cls * 37) % 256);
            uint8_t g = static_cast<uint8_t>((cls * 91) % 256);
            uint8_t b = static_cast<uint8_t>((cls * 157) % 256);
            *iter_rgb = (static_cast<uint32_t>(r) << 16) | 
                        (static_cast<uint32_t>(g) << 8) | b;
            
            ++iter_x; ++iter_y; ++iter_z; ++iter_rgb;
        }
        
        map_pub_.publish(msg);
    }
    
    void publishCostmap() {
        if (costmap_pub_.getNumSubscribers() == 0) return;
        
        boost::mutex::scoped_lock lock(pipeline_mutex_);
        
        int width, height;
        auto data = pipeline_->getCostmap(width, height);
        if (data.empty()) return;
        
        nav_msgs::OccupancyGrid msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = map_frame_;
        msg.info.resolution = config_.map.resolution;
        msg.info.width = width;
        msg.info.height = height;
        msg.info.origin.position.x = config_.map.origin_x;
        msg.info.origin.position.y = config_.map.origin_y;
        msg.info.origin.orientation.w = 1.0;
        msg.data = data;
        
        costmap_pub_.publish(msg);
    }
    
    void publishPrimitives() {
        if (primitives_pub_.getNumSubscribers() == 0) return;
        
        boost::mutex::scoped_lock lock(pipeline_mutex_);
        
        visualization_msgs::MarkerArray markers;
        
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
            
            marker.pose.position.x = prim.centroid.x();
            marker.pose.position.y = prim.centroid.y();
            marker.pose.position.z = prim.centroid.z();
            
            auto orientation = prim.getOrientation();
            marker.pose.orientation.x = orientation.x();
            marker.pose.orientation.y = orientation.y();
            marker.pose.orientation.z = orientation.z();
            marker.pose.orientation.w = orientation.w();
            
            auto eigenvalues = prim.getEigenvalues();
            marker.scale.x = 2.0 * std::sqrt(std::max(0.01, eigenvalues(0)));
            marker.scale.y = 2.0 * std::sqrt(std::max(0.01, eigenvalues(1)));
            marker.scale.z = 2.0 * std::sqrt(std::max(0.01, eigenvalues(2)));
            
            marker.color.r = prim.uncertainty;
            marker.color.g = 1.0 - prim.uncertainty;
            marker.color.b = 0.0;
            marker.color.a = 0.5;
            marker.lifetime = ros::Duration(0.5);
            
            markers.markers.push_back(marker);
        }
        
        primitives_pub_.publish(markers);
    }
    
    // ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Publishers/Subscribers
    ros::Publisher map_pub_;
    ros::Publisher costmap_pub_;
    ros::Publisher primitives_pub_;
    ros::Publisher uncertainty_map_pub_;
    ros::Subscriber cloud_sub_;
    ros::ServiceServer reset_srv_;
    ros::Timer map_timer_;
    ros::Timer costmap_timer_;
    
    // HESFM
    HESFMConfig config_;
    std::unique_ptr<HESFMPipeline> pipeline_;
    std::vector<GaussianPrimitive> current_primitives_;
    boost::mutex pipeline_mutex_;
    
    // Async processing
    bool async_processing_ = false;
    bool running_ = false;
    boost::thread processing_thread_;
    std::queue<sensor_msgs::PointCloud2::ConstPtr> cloud_queue_;
    boost::mutex queue_mutex_;
    boost::condition_variable queue_condition_;
    
    // Parameters
    std::string map_frame_;
    std::string sensor_frame_;
    std::vector<int> traversable_classes_;
};

} // namespace hesfm

PLUGINLIB_EXPORT_CLASS(hesfm::HESFMMapperNodelet, nodelet::Nodelet)
