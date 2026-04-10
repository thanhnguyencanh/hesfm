/**
 * @file hesfm_mapper_nodelet.cpp
 * @brief HESFM Mapper Nodelet for zero-copy data transfer
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 *
 * Synced with hesfm_mapper_node.cpp:
 *   - YAML-nested parameter loading with flat launch-arg overrides
 *   - Compact class color palette from relevant_classes
 *   - Full convertPointCloud (RGB, uncertainty_semantic, is_traversable)
 *   - getLastPrimitives() instead of recomputing
 *   - Proper class-color semantic map publishing
 *   - Uncertainty map + uncertainty cloud publishers
 *   - Processing statistics
 *
 * Nodelet-specific:
 *   - shared_ptr TF buffer/listener (no member-order dependency)
 *   - boost::mutex for pipeline thread safety
 *   - Optional async processing thread with queue
 *   - Zero-copy PointCloud2 via ConstPtr within nodelet manager
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <boost/thread.hpp>
#include <algorithm>
#include <cstring>
#include <memory>
#include <queue>
#include <set>

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
        NODELET_INFO("  Num classes: %d", config_.map.num_classes);
    }

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
        const auto& dataset_colors = (dataset == "nyuv2") ? NYUV2_CLASS_COLORS
                                    : SUNRGBD_CLASS_COLORS;

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
        pnh_.param("primitives/conflict_threshold",       config_.primitive.conflict_threshold,        0.7);

        // Kernel parameters — YAML nested under "kernel/"
        pnh_.param("kernel/length_scale_min",      config_.kernel.length_scale_min,      0.1);
        pnh_.param("kernel/length_scale_max",      config_.kernel.length_scale_max,      0.5);
        pnh_.param("kernel/uncertainty_threshold", config_.kernel.uncertainty_threshold,  0.75);
        pnh_.param("kernel/gamma",                 config_.kernel.gamma,                  2.0);

        // Processing parameters — YAML nested under "processing/"
        pnh_.param("processing/map_publish_rate",     config_.processing.map_publish_rate,     2.0);

        // Async processing (nodelet-specific)
        pnh_.param("async_processing", async_processing_, false);

        // Traversable classes
        std::vector<int> traversable_default = {1, 19};  // floor, floor_mat (SUNRGBD)
        pnh_.param("navigation/traversable_classes", traversable_classes_, traversable_default);

        // Relevant classes filter — empty means map all classes
        std::vector<int> relevant_default = {};
        pnh_.param("navigation/relevant_classes", relevant_classes_, relevant_default);
        relevant_set_.insert(relevant_classes_.begin(), relevant_classes_.end());

        // Build compact class color palette when remapping is enabled
        if (!relevant_classes_.empty()) {
            std::vector<int> sorted_relevant = relevant_classes_;
            std::sort(sorted_relevant.begin(), sorted_relevant.end());

            class_colors_.clear();
            class_colors_.reserve(sorted_relevant.size() + 1);
            for (int old_idx : sorted_relevant) {
                if (old_idx >= 0 && old_idx < static_cast<int>(dataset_colors.size())) {
                    class_colors_.push_back(dataset_colors[old_idx]);
                } else {
                    class_colors_.push_back({255, 165, 0});
                }
            }
            class_colors_.push_back({255, 165, 0});  // other
        } else {
            class_colors_ = dataset_colors;
        }

        // Optional visualization outputs (default off — enable via launch arg)
        pnh_.param("publish_primitives",        publish_primitives_,       false);
        pnh_.param("publish_uncertainty_map",   publish_uncertainty_map_,  false);
        pnh_.param("publish_uncertainty_cloud", publish_uncertainty_cloud_, false);
    }

    void initializeHESFM() {
        pipeline_ = std::make_unique<HESFMPipeline>(config_);

        std::set<int> trav_set(traversable_classes_.begin(), traversable_classes_.end());
        pipeline_->getMap().setTraversableClasses(trav_set);
        pipeline_->getKernel().setTraversableClasses(trav_set);

        NODELET_INFO("HESFM pipeline initialized");
    }

    void setupSubscribers() {
        cloud_sub_ = nh_.subscribe("semantic_cloud", 1,
                                    &HESFMMapperNodelet::cloudCallback, this);

        NODELET_INFO("Subscribed to: %s", cloud_sub_.getTopic().c_str());
    }

    void setupPublishers() {
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_map", 1);

        if (publish_primitives_)
            primitives_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("primitives", 1);

        if (publish_uncertainty_map_)
            uncertainty_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("uncertainty_map", 1);

        if (publish_uncertainty_cloud_)
            uncertainty_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("uncertainty_cloud", 1);

        NODELET_INFO("Publishers initialized (primitives=%s unc_map=%s unc_cloud=%s)",
                     publish_primitives_ ? "on" : "off",
                     publish_uncertainty_map_ ? "on" : "off",
                     publish_uncertainty_cloud_ ? "on" : "off");
    }

    void setupServices() {
        reset_srv_ = nh_.advertiseService("reset_map",
                                           &HESFMMapperNodelet::resetMapCallback, this);

        NODELET_INFO("Services initialized");
    }

    void setupTimers() {
        map_timer_ = nh_.createTimer(
            ros::Duration(1.0 / config_.processing.map_publish_rate),
            &HESFMMapperNodelet::mapPublishCallback, this);
    }

    // =========================================================================
    // Callbacks
    // =========================================================================

    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (async_processing_) {
            boost::mutex::scoped_lock lock(queue_mutex_);
            cloud_queue_.push(msg);
            queue_condition_.notify_one();
        } else {
            processCloud(msg);
        }
    }

    void processCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        ros::Time start_time = ros::Time::now();

        // Get transform from sensor to map frame
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(
                map_frame_, msg->header.frame_id,
                msg->header.stamp, ros::Duration(0.1));
        } catch (tf2::TransformException& ex) {
            NODELET_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return;
        }

        // Extract sensor origin
        Vector3d sensor_origin(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z);

        // Convert point cloud to HESFM format
        std::vector<SemanticPoint> points;
        if (!convertPointCloud(msg, transform, points)) {
            NODELET_WARN_THROTTLE(1.0, "Failed to convert point cloud");
            return;
        }

        if (points.empty()) return;

        // Process through HESFM pipeline
        int num_primitives;
        {
            boost::mutex::scoped_lock lock(pipeline_mutex_);
            num_primitives = pipeline_->process(points, sensor_origin);
            if (publish_primitives_)
                current_primitives_ = pipeline_->getLastPrimitives();
        }

        // Publish per-point uncertainty decomposition cloud
        if (publish_uncertainty_cloud_ && uncertainty_cloud_pub_.getNumSubscribers() > 0) {
            publishUncertaintyCloud(points, msg->header);
        }

        // Update timing statistics
        double processing_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        updateStatistics(processing_time, points.size(), num_primitives);

        NODELET_DEBUG("Processed %zu points, %d primitives in %.1f ms",
                      points.size(), num_primitives, processing_time);
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

    void mapPublishCallback(const ros::TimerEvent&) {
        // Snapshot once under lock — shared by all publish methods needing cell data.
        std::vector<MapCell> cells;
        {
            const bool need_cells = map_pub_.getNumSubscribers() > 0
                || (publish_uncertainty_map_ && uncertainty_map_pub_.getNumSubscribers() > 0);
            if (need_cells) {
                boost::mutex::scoped_lock lock(pipeline_mutex_);
                cells = pipeline_->getMap().getOccupiedCells();
            }
        }

        publishSemanticMap(cells);
        if (publish_primitives_)      publishPrimitives();
        if (publish_uncertainty_map_) publishUncertaintyMap(cells);
    }

    bool resetMapCallback(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {
        boost::mutex::scoped_lock lock(pipeline_mutex_);
        pipeline_->reset();
        NODELET_INFO("Map reset");
        return true;
    }

    // =========================================================================
    // Point Cloud Conversion
    // =========================================================================

    bool convertPointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg,
                           const geometry_msgs::TransformStamped& transform,
                           std::vector<SemanticPoint>& points) {

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
            NODELET_ERROR("Point cloud missing xyz fields");
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

            SemanticPoint sp;
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

    void publishSemanticMap(const std::vector<MapCell>& cells) {
        if (map_pub_.getNumSubscribers() == 0 || cells.empty()) return;

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

    void publishPrimitives() {
        if (primitives_pub_.getNumSubscribers() == 0) return;

        boost::mutex::scoped_lock lock(pipeline_mutex_);

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

    void publishUncertaintyMap(const std::vector<MapCell>& cells) {
        if (uncertainty_map_pub_.getNumSubscribers() == 0 || cells.empty()) return;

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

    /**
     * @brief Publish per-point uncertainty decomposition as a PointCloud2
     *
     * Each point carries 5 intensity channels (u_sem, u_spa, u_obs, u_temp, u_total).
     * Fields: x, y, z, u_sem, u_spa, u_obs, u_temp, u_total.
     */
    void publishUncertaintyCloud(const std::vector<SemanticPoint>& points,
                                  const std_msgs::Header& header) {

        // Downsample: publish every 4th point to keep bandwidth low
        const int step = 4;
        size_t out_count = (points.size() + step - 1) / step;

        sensor_msgs::PointCloud2 msg;
        msg.header = header;
        msg.header.frame_id = map_frame_;
        msg.height = 1;
        msg.width = static_cast<uint32_t>(out_count);
        msg.is_bigendian = false;
        msg.is_dense = true;

        // Build fields manually (8 x float32 = 32 bytes per point)
        const uint32_t point_step = 8 * sizeof(float);
        const char* names[] = {"x","y","z","u_sem","u_spa","u_obs","u_temp","u_total"};
        msg.fields.resize(8);
        for (int f = 0; f < 8; ++f) {
            msg.fields[f].name     = names[f];
            msg.fields[f].offset   = f * sizeof(float);
            msg.fields[f].datatype = sensor_msgs::PointField::FLOAT32;
            msg.fields[f].count    = 1;
        }
        msg.point_step = point_step;
        msg.row_step   = point_step * msg.width;
        msg.data.resize(msg.row_step * msg.height);

        size_t idx = 0;
        for (size_t i = 0; i < points.size(); i += step, ++idx) {
            const auto& p = points[i];
            float vals[8] = {
                static_cast<float>(p.position.x()),
                static_cast<float>(p.position.y()),
                static_cast<float>(p.position.z()),
                static_cast<float>(p.uncertainty_semantic),
                static_cast<float>(p.uncertainty_spatial),
                static_cast<float>(p.uncertainty_observation),
                static_cast<float>(p.uncertainty_temporal),
                static_cast<float>(p.uncertainty_total)
            };
            memcpy(&msg.data[idx * point_step], vals, point_step);
        }

        uncertainty_cloud_pub_.publish(msg);
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
            NODELET_INFO("HESFM Stats: %.1f FPS, %zu cells, %.1f%% coverage",
                         fps, pipeline_->getMap().getNumCells(),
                         pipeline_->getMap().getCoverage() * 100.0);
        }
    }

    // =========================================================================
    // Member Variables
    // =========================================================================

    // ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publishers
    ros::Publisher map_pub_;
    ros::Publisher primitives_pub_;
    ros::Publisher uncertainty_map_pub_;
    ros::Publisher uncertainty_cloud_pub_;

    // Subscribers
    ros::Subscriber cloud_sub_;

    // Services
    ros::ServiceServer reset_srv_;

    // Timers
    ros::Timer map_timer_;

    // HESFM
    HESFMConfig config_;
    std::unique_ptr<HESFMPipeline> pipeline_;
    std::vector<GaussianPrimitive> current_primitives_;
    boost::mutex pipeline_mutex_;

    // Async processing (nodelet-specific)
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
    std::vector<int> relevant_classes_;
    std::set<int> relevant_set_;
    std::vector<std::array<uint8_t, 3>> class_colors_;

    // Visualization toggles
    bool publish_primitives_        = false;
    bool publish_uncertainty_map_   = false;
    bool publish_uncertainty_cloud_ = false;

    // Statistics
    int total_frames_ = 0;
    size_t total_points_ = 0;
    double total_processing_time_ = 0.0;
};

} // namespace hesfm

PLUGINLIB_EXPORT_CLASS(hesfm::HESFMMapperNodelet, nodelet::Nodelet)
