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


#include <boost/thread.hpp>
#include <algorithm>
#include <cstring>
#include <memory>
#include <atomic>
#include <set>
#include <unordered_set>

#include "hesfm/hesfm.h"

namespace {

template <typename T>
void loadParamWithLegacyFallback(ros::NodeHandle& pnh,
                                 const std::string& nested_name,
                                 const std::string& legacy_name,
                                 T& value,
                                 const T& default_value) {
    pnh.param(nested_name, value, default_value);
    pnh.param(legacy_name, value, value);
}

bool getParamWithLegacyFallback(const ros::NodeHandle& pnh,
                                const std::string& nested_name,
                                const std::string& legacy_name,
                                std::vector<int>& value) {
    return pnh.getParam(nested_name, value) || pnh.getParam(legacy_name, value);
}

}  // namespace

// Mirror of the packed struct written by semantic_cloud_nodelet / semantic_cloud_node.
// Must stay byte-for-byte identical to SemanticPoint in those files.
struct IncomingPoint {
    float    x, y, z;
    uint32_t rgb;
    uint32_t label;
    float    uncertainty;
};
static_assert(sizeof(IncomingPoint) == 24, "IncomingPoint layout mismatch");

static bool isKnownLayout(const sensor_msgs::PointCloud2& msg) {
    if (msg.point_step != 24u) return false;  // sizeof(IncomingPoint)
    if (msg.fields.size() < 6) return false;
    for (const auto& f : msg.fields) {
        uint32_t expected_offset = 0;
        uint8_t  expected_type   = 0;
        if      (f.name == "x")           { expected_offset =  0; expected_type = sensor_msgs::PointField::FLOAT32; }
        else if (f.name == "y")           { expected_offset =  4; expected_type = sensor_msgs::PointField::FLOAT32; }
        else if (f.name == "z")           { expected_offset =  8; expected_type = sensor_msgs::PointField::FLOAT32; }
        else if (f.name == "rgb")         { expected_offset = 12; expected_type = sensor_msgs::PointField::UINT32;  }
        else if (f.name == "label")       { expected_offset = 16; expected_type = sensor_msgs::PointField::UINT32;  }
        else if (f.name == "uncertainty") { expected_offset = 20; expected_type = sensor_msgs::PointField::FLOAT32; }
        else continue;
        if (f.offset != expected_offset || f.datatype != expected_type) return false;
    }
    return true;
}

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
        loadParamWithLegacyFallback(
            pnh_, "map/resolution", "resolution", config_.map.resolution, config_.map.resolution);
        loadParamWithLegacyFallback(
            pnh_, "map/num_classes", "num_classes", config_.map.num_classes, config_.map.num_classes);
        loadParamWithLegacyFallback(
            pnh_, "map/size_x", "map_size_x", config_.map.size_x, config_.map.size_x);
        loadParamWithLegacyFallback(
            pnh_, "map/size_y", "map_size_y", config_.map.size_y, config_.map.size_y);
        loadParamWithLegacyFallback(
            pnh_, "map/size_z", "map_size_z", config_.map.size_z, config_.map.size_z);
        loadParamWithLegacyFallback(
            pnh_, "map/origin_x", "origin_x", config_.map.origin_x, config_.map.origin_x);
        loadParamWithLegacyFallback(
            pnh_, "map/origin_y", "origin_y", config_.map.origin_y, config_.map.origin_y);
        loadParamWithLegacyFallback(
            pnh_, "map/origin_z", "origin_z", config_.map.origin_z, config_.map.origin_z);
        loadParamWithLegacyFallback(
            pnh_, "map/log_odds_min", "log_odds_min", config_.map.log_odds_min, config_.map.log_odds_min);
        loadParamWithLegacyFallback(
            pnh_, "map/log_odds_max", "log_odds_max", config_.map.log_odds_max, config_.map.log_odds_max);
        config_.map.prior_prob = 1.0 / static_cast<double>(std::max(config_.map.num_classes, 1));
        loadParamWithLegacyFallback(
            pnh_, "map/prior_prob", "prior_prob", config_.map.prior_prob, config_.map.prior_prob);
        int map_max_cells = static_cast<int>(config_.map.max_cells);
        loadParamWithLegacyFallback(
            pnh_, "map/max_cells", "max_cells", map_max_cells, map_max_cells);
        config_.map.max_cells = map_max_cells > 0 ? static_cast<size_t>(map_max_cells) : 0u;

        // Uncertainty weights — YAML nested under "uncertainty/"
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/w_semantic",
            "w_semantic",
            config_.uncertainty.w_semantic,
            config_.uncertainty.w_semantic);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/w_spatial",
            "w_spatial",
            config_.uncertainty.w_spatial,
            config_.uncertainty.w_spatial);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/w_observation",
            "w_observation",
            config_.uncertainty.w_observation,
            config_.uncertainty.w_observation);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/w_temporal",
            "w_temporal",
            config_.uncertainty.w_temporal,
            config_.uncertainty.w_temporal);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/spatial_radius",
            "spatial_radius",
            config_.uncertainty.spatial_radius,
            config_.uncertainty.spatial_radius);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/min_neighbors",
            "min_neighbors",
            config_.uncertainty.min_neighbors,
            config_.uncertainty.min_neighbors);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/sigma_range",
            "sigma_range",
            config_.uncertainty.sigma_range,
            config_.uncertainty.sigma_range);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/sigma_density",
            "sigma_density",
            config_.uncertainty.sigma_density,
            config_.uncertainty.sigma_density);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/sigma_angle",
            "sigma_angle",
            config_.uncertainty.sigma_angle,
            config_.uncertainty.sigma_angle);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/max_density",
            "max_density",
            config_.uncertainty.max_density,
            config_.uncertainty.max_density);
        loadParamWithLegacyFallback(
            pnh_,
            "uncertainty/temporal_resolution",
            "temporal_resolution",
            config_.uncertainty.temporal_resolution,
            config_.uncertainty.temporal_resolution);
        config_.uncertainty.normalizeWeights();

        // Gaussian primitive parameters — YAML nested under "primitives/"
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/target_primitives",
            "target_primitives",
            config_.primitive.target_primitives,
            config_.primitive.target_primitives);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/min_points_per_primitive",
            "min_points_per_primitive",
            config_.primitive.min_points_per_primitive,
            config_.primitive.min_points_per_primitive);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/max_points_per_primitive",
            "max_points_per_primitive",
            config_.primitive.max_points_per_primitive,
            config_.primitive.max_points_per_primitive);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/regularization",
            "regularization",
            config_.primitive.regularization,
            config_.primitive.regularization);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/uncertainty_weight_lambda",
            "uncertainty_weight_lambda",
            config_.primitive.uncertainty_weight_lambda,
            config_.primitive.uncertainty_weight_lambda);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/kmeans_max_iter",
            "kmeans_max_iter",
            config_.primitive.kmeans_max_iter,
            config_.primitive.kmeans_max_iter);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/kmeans_tolerance",
            "kmeans_tolerance",
            config_.primitive.kmeans_tolerance,
            config_.primitive.kmeans_tolerance);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/conflict_threshold",
            "conflict_threshold",
            config_.primitive.conflict_threshold,
            config_.primitive.conflict_threshold);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/use_incremental_updates",
            "use_incremental_updates",
            config_.primitive.use_incremental_updates,
            config_.primitive.use_incremental_updates);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/incremental_max_distance",
            "incremental_max_distance",
            config_.primitive.incremental_max_distance,
            config_.primitive.incremental_max_distance);
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/full_rebuild_interval",
            "full_rebuild_interval",
            config_.primitive.full_rebuild_interval,
            config_.primitive.full_rebuild_interval);
        config_.primitive.num_classes = config_.map.num_classes;
        loadParamWithLegacyFallback(
            pnh_,
            "primitives/num_classes",
            "primitive_num_classes",
            config_.primitive.num_classes,
            config_.primitive.num_classes);

        // Kernel parameters — YAML nested under "kernel/"
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/length_scale_min",
            "length_scale_min",
            config_.kernel.length_scale_min,
            config_.kernel.length_scale_min);
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/length_scale_max",
            "length_scale_max",
            config_.kernel.length_scale_max,
            config_.kernel.length_scale_max);
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/uncertainty_threshold",
            "uncertainty_threshold",
            config_.kernel.uncertainty_threshold,
            config_.kernel.uncertainty_threshold);
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/uncertainty_low",
            "uncertainty_low",
            config_.kernel.uncertainty_low,
            config_.kernel.uncertainty_low);
        loadParamWithLegacyFallback(
            pnh_, "kernel/gamma", "gamma", config_.kernel.gamma, config_.kernel.gamma);
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/reachability_lambda",
            "reachability_lambda",
            config_.kernel.reachability_lambda,
            config_.kernel.reachability_lambda);
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/confidence_weight_beta",
            "confidence_weight_beta",
            config_.kernel.confidence_weight_beta,
            config_.kernel.confidence_weight_beta);
        loadParamWithLegacyFallback(
            pnh_,
            "kernel/entropy_weight_gamma",
            "entropy_weight_gamma",
            config_.kernel.entropy_weight_gamma,
            config_.kernel.entropy_weight_gamma);

        // Processing parameters — YAML nested under "processing/"
        loadParamWithLegacyFallback(
            pnh_,
            "processing/map_publish_rate",
            "map_publish_rate",
            config_.processing.map_publish_rate,
            config_.processing.map_publish_rate);

        // Async processing (nodelet-specific)
        config_.processing.use_async_processing = false;
        loadParamWithLegacyFallback(
            pnh_,
            "processing/use_async_processing",
            "async_processing",
            config_.processing.use_async_processing,
            false);
        async_processing_ = config_.processing.use_async_processing;

        // Sensor parameters used by uncertainty decomposition and cloud filtering
        loadParamWithLegacyFallback(
            pnh_,
            "sensor/min_range",
            "min_range",
            config_.processing.sensor_model.min_range,
            config_.processing.sensor_model.min_range);
        loadParamWithLegacyFallback(
            pnh_,
            "sensor/max_range",
            "max_range",
            config_.processing.sensor_model.max_range,
            config_.processing.sensor_model.max_range);

        // Traversable classes
        std::vector<int> traversable_default = {1, 19};  // floor, floor_mat (SUNRGBD)
        if (!getParamWithLegacyFallback(
                pnh_,
                "navigation/traversable_classes",
                "traversable_classes",
                traversable_classes_)) {
            traversable_classes_ = traversable_default;
        }

        // Relevant classes filter — empty means map all classes
        std::vector<int> relevant_default = {};
        if (!getParamWithLegacyFallback(
                pnh_,
                "navigation/relevant_classes",
                "relevant_classes",
                relevant_classes_)) {
            relevant_classes_ = relevant_default;
        }
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
        traversable_set_ = std::unordered_set<int>(traversable_classes_.begin(), traversable_classes_.end());
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
            if (latest_cloud_) ++dropped_frames_;   // overwrite means previous frame is dropped
            latest_cloud_ = msg;
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
                while (!latest_cloud_ && running_) {
                    queue_condition_.timed_wait(lock, boost::posix_time::milliseconds(100));
                }

                if (!running_) break;

                msg = std::move(latest_cloud_);   // take ownership; slot is now empty
            }

            if (dropped_frames_ > 0) {
                NODELET_WARN_THROTTLE(2.0, "[mapper] dropped %u frame(s) — mapper behind camera",
                                      dropped_frames_.load());
                dropped_frames_ = 0;
            }

            processCloud(msg);
        }
    }

    void mapPublishCallback(const ros::TimerEvent&) {
        // Snapshot once under lock — shared by all publish methods needing cell data.
        std::vector<SemanticMap::CellView> cells;
        {
            const bool need_cells = map_pub_.getNumSubscribers() > 0
                || (publish_uncertainty_map_ && uncertainty_map_pub_.getNumSubscribers() > 0);
            if (need_cells) {
                boost::mutex::scoped_lock lock(pipeline_mutex_);
                cells = pipeline_->getMap().getCellViews();
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

        const uint32_t N = msg->width * msg->height;
        if (N == 0) return true;

        // ── Fast path: known packed layout from our own cloud nodelet ───────────
        if (!cloud_layout_checked_) {
            cloud_layout_fast_ = isKnownLayout(*msg);
            cloud_layout_checked_ = true;
            NODELET_INFO("[mapper] cloud layout: %s",
                         cloud_layout_fast_ ? "fast (direct cast)" : "generic (field scan)");
        }

        const Eigen::Affine3d tf_eigen = tf2::transformToEigen(transform);
        points.reserve(N);

        if (cloud_layout_fast_) {
            const IncomingPoint* in =
                reinterpret_cast<const IncomingPoint*>(msg->data.data());

            for (uint32_t i = 0; i < N; ++i) {
                const IncomingPoint& p = in[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                    continue;

                SemanticPoint sp;
                sp.position             = tf_eigen * Eigen::Vector3d(p.x, p.y, p.z);
                sp.depth                = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
                sp.semantic_class       = static_cast<int>(p.label);
                sp.semantic_confidence  = 1.0f - p.uncertainty;
                sp.uncertainty_semantic = p.uncertainty;
                sp.uncertainty_total    = p.uncertainty;
                sp.r = (p.rgb >> 16) & 0xFF;
                sp.g = (p.rgb >>  8) & 0xFF;
                sp.b =  p.rgb        & 0xFF;
                sp.is_traversable = traversable_set_.count(sp.semantic_class) > 0;
                points.push_back(sp);
            }
            return true;
        }

        // ── Generic fallback: scan fields once, then parse per-point ────────────
        int x_idx = -1, y_idx = -1, z_idx = -1;
        int label_idx = -1, uncertainty_idx = -1, rgb_idx = -1;
        for (size_t fi = 0; fi < msg->fields.size(); ++fi) {
            const auto& f = msg->fields[fi];
            if      (f.name == "x")           x_idx           = fi;
            else if (f.name == "y")           y_idx           = fi;
            else if (f.name == "z")           z_idx           = fi;
            else if (f.name == "label")       label_idx       = fi;
            else if (f.name == "uncertainty") uncertainty_idx = fi;
            else if (f.name == "rgb")         rgb_idx         = fi;
        }
        if (x_idx < 0 || y_idx < 0 || z_idx < 0) {
            NODELET_ERROR("Point cloud missing xyz fields");
            return false;
        }

        const uint32_t step = msg->point_step;
        for (uint32_t i = 0; i < N; ++i) {
            const uint8_t* ptr = msg->data.data() + i * step;
            float x, y, z;
            memcpy(&x, ptr + msg->fields[x_idx].offset, sizeof(float));
            memcpy(&y, ptr + msg->fields[y_idx].offset, sizeof(float));
            memcpy(&z, ptr + msg->fields[z_idx].offset, sizeof(float));
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            SemanticPoint sp;
            sp.position = tf_eigen * Eigen::Vector3d(x, y, z);
            sp.depth    = std::sqrt(x*x + y*y + z*z);

            if (label_idx >= 0) {
                uint32_t label; memcpy(&label, ptr + msg->fields[label_idx].offset, sizeof(uint32_t));
                sp.semantic_class = static_cast<int>(label);
            }
            if (uncertainty_idx >= 0) {
                float unc; memcpy(&unc, ptr + msg->fields[uncertainty_idx].offset, sizeof(float));
                sp.uncertainty_semantic = unc;
                sp.uncertainty_total    = unc;
            } else {
                sp.uncertainty_semantic = 0.3f;
                sp.uncertainty_total    = 0.3f;
            }
            if (rgb_idx >= 0) {
                uint32_t rgb; memcpy(&rgb, ptr + msg->fields[rgb_idx].offset, sizeof(uint32_t));
                sp.r = (rgb >> 16) & 0xFF;
                sp.g = (rgb >>  8) & 0xFF;
                sp.b =  rgb        & 0xFF;
            }
            sp.is_traversable = traversable_set_.count(sp.semantic_class) > 0;
            points.push_back(sp);
        }
        return true;
    }

    // =========================================================================
    // Publishing
    // =========================================================================

    void publishSemanticMap(const std::vector<SemanticMap::CellView>& cells) {
        if (map_pub_.getNumSubscribers() == 0 || cells.empty()) return;

        struct MapPoint { float x, y, z; uint32_t rgb; };
        static_assert(sizeof(MapPoint) == 16, "");

        sensor_msgs::PointCloud2 msg;
        msg.header.stamp    = ros::Time::now();
        msg.header.frame_id = map_frame_;
        msg.height      = 1;
        msg.is_bigendian = false;
        msg.is_dense     = true;
        msg.point_step   = sizeof(MapPoint);
        msg.fields.resize(4);
        {
            auto mk = [](const char* n, uint32_t off, uint8_t dt) {
                sensor_msgs::PointField f; f.name=n; f.offset=off; f.datatype=dt; f.count=1; return f;
            };
            msg.fields[0] = mk("x",   0, sensor_msgs::PointField::FLOAT32);
            msg.fields[1] = mk("y",   4, sensor_msgs::PointField::FLOAT32);
            msg.fields[2] = mk("z",   8, sensor_msgs::PointField::FLOAT32);
            msg.fields[3] = mk("rgb",12, sensor_msgs::PointField::UINT32);
        }

        msg.data.resize(cells.size() * sizeof(MapPoint));
        MapPoint* out = reinterpret_cast<MapPoint*>(msg.data.data());
        const int ncol = static_cast<int>(class_colors_.size());

        for (size_t i = 0; i < cells.size(); ++i) {
            out[i].x = cells[i].x;
            out[i].y = cells[i].y;
            out[i].z = cells[i].z;
            const int cls = static_cast<int>(cells[i].pred_class);
            if (cls >= 0 && cls < ncol) {
                const auto& c = class_colors_[cls];
                out[i].rgb = (uint32_t(c[0]) << 16) | (uint32_t(c[1]) << 8) | c[2];
            } else {
                out[i].rgb = 0x808080u;
            }
        }

        msg.width    = static_cast<uint32_t>(cells.size());
        msg.row_step = msg.point_step * msg.width;
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

    void publishUncertaintyMap(const std::vector<SemanticMap::CellView>& cells) {
        if (uncertainty_map_pub_.getNumSubscribers() == 0 || cells.empty()) return;

        struct UncPoint { float x, y, z, intensity; };
        static_assert(sizeof(UncPoint) == 16, "");

        sensor_msgs::PointCloud2 msg;
        msg.header.stamp    = ros::Time::now();
        msg.header.frame_id = map_frame_;
        msg.height      = 1;
        msg.is_bigendian = false;
        msg.is_dense     = true;
        msg.point_step   = sizeof(UncPoint);
        msg.fields.resize(4);
        {
            auto mk = [](const char* n, uint32_t off) {
                sensor_msgs::PointField f; f.name=n; f.offset=off;
                f.datatype=sensor_msgs::PointField::FLOAT32; f.count=1; return f;
            };
            msg.fields[0] = mk("x",         0);
            msg.fields[1] = mk("y",         4);
            msg.fields[2] = mk("z",         8);
            msg.fields[3] = mk("intensity", 12);
        }

        msg.data.resize(cells.size() * sizeof(UncPoint));
        UncPoint* out = reinterpret_cast<UncPoint*>(msg.data.data());

        for (size_t i = 0; i < cells.size(); ++i) {
            out[i].x         = cells[i].x;
            out[i].y         = cells[i].y;
            out[i].z         = cells[i].z;
            out[i].intensity = 1.0f - cells[i].confidence;
        }

        msg.width    = static_cast<uint32_t>(cells.size());
        msg.row_step = msg.point_step * msg.width;
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
    sensor_msgs::PointCloud2::ConstPtr latest_cloud_;   // "queue of 1" — overwritten on each arrival
    boost::mutex queue_mutex_;
    boost::condition_variable queue_condition_;
    std::atomic<uint32_t> dropped_frames_{0};

    // Parameters
    std::string map_frame_;
    std::string sensor_frame_;
    std::vector<int> traversable_classes_;
    std::unordered_set<int> traversable_set_;   // O(1) lookup in convertPointCloud
    std::vector<int> relevant_classes_;
    std::set<int> relevant_set_;
    std::vector<std::array<uint8_t, 3>> class_colors_;

    // Cloud layout detection (checked once on first message)
    bool cloud_layout_checked_ = false;
    bool cloud_layout_fast_    = false;

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
