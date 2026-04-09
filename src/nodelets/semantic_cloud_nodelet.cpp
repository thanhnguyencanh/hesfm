/**
 * @file semantic_cloud_nodelet.cpp
 * @brief Semantic Point Cloud Generator Nodelet
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 *
 * Generates semantic point clouds from depth + segmentation images
 * with zero-copy data transfer when used with other nodelets.
 *
 * Synced with semantic_cloud_node.cpp optimizations:
 *   - No RGB subscription (colors from class palette)
 *   - Packed SemanticPoint struct, direct write into cloud_msg.data
 *   - 3-topic / 2-topic sync (depth + semantic [+ uncertainty])
 *   - Compact palette remap from relevant_classes
 *   - Raw row pointers for semantic/uncertainty access
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>

#include "hesfm/types.h"

namespace hesfm {

// Packed point layout — identical to semantic_cloud_node.cpp
struct SemanticPointPacked {
    float    x, y, z;
    uint32_t rgb;
    uint32_t label;
    float    uncertainty;
};

class SemanticCloudNodelet : public nodelet::Nodelet {
public:
    SemanticCloudNodelet() = default;
    virtual ~SemanticCloudNodelet() = default;

private:
    virtual void onInit() override {
        nh_  = getNodeHandle();
        pnh_ = getPrivateNodeHandle();

        loadParameters();
        setupPublishers();
        setupSubscribers();

        NODELET_INFO("Semantic Cloud Nodelet initialized");
        NODELET_INFO("  Dataset: %s (%d classes)", dataset_.c_str(), num_classes_);
        NODELET_INFO("  Downsample factor: %d", downsample_factor_);
        NODELET_INFO("  Depth range: [%.2f, %.2f] m", min_depth_, max_depth_);
        NODELET_INFO("  Output frame: %s",
                     output_frame_.empty() ? "(camera frame)" : output_frame_.c_str());
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    void loadParameters() {
        pnh_.param<std::string>("esanet_dataset", dataset_, "nyuv2");
        if (dataset_ == "sunrgbd") {
            num_classes_  = SUNRGBD_NUM_CLASSES;
            class_colors_ = SUNRGBD_CLASS_COLORS;
        } else {
            num_classes_  = NYUV2_NUM_CLASSES;
            class_colors_ = NYUV2_CLASS_COLORS;
        }
        base_class_colors_ = class_colors_;
        pnh_.param("num_classes", num_classes_, num_classes_);

        pnh_.param("downsample_factor",   downsample_factor_, 2);
        pnh_.param("min_depth",           min_depth_,         0.06);
        pnh_.param("max_depth",           max_depth_,         6.0);
        pnh_.param("queue_size",          queue_size_,        10);
        pnh_.param("use_uncertainty",     use_uncertainty_,   true);
        pnh_.param("default_uncertainty", default_uncertainty_, 0.3f);
        pnh_.param<std::string>("output_frame", output_frame_, "");

        tryConfigureCompactPalette();
    }

    bool tryConfigureCompactPalette() {
        if (compact_palette_configured_) return true;

        std::vector<int> relevant;
        if (!pnh_.getParam("relevant_classes", relevant))
            nh_.getParam("/hesfm_mapper_node/navigation/relevant_classes", relevant);
        if (relevant.empty()) return false;

        std::sort(relevant.begin(), relevant.end());
        relevant.erase(std::unique(relevant.begin(), relevant.end()), relevant.end());

        const std::array<uint8_t, 3> other_color = {255, 165, 0};
        compact_colors_.clear();
        compact_colors_.reserve(relevant.size() + 1);
        for (int idx : relevant) {
            if (idx >= 0 && idx < static_cast<int>(base_class_colors_.size()))
                compact_colors_.push_back(base_class_colors_[idx]);
            else
                compact_colors_.push_back(other_color);
        }
        compact_colors_.push_back(other_color);

        relevant_classes_ = relevant;
        class_colors_     = compact_colors_;
        num_classes_      = static_cast<int>(relevant.size()) + 1;
        compact_palette_configured_ = true;

        NODELET_INFO("  Compact palette: %d classes (%zu relevant + other)",
                     num_classes_, relevant.size());
        return true;
    }

    void setupSubscribers() {
        camera_info_sub_ = nh_.subscribe("color/camera_info", 1,
                                          &SemanticCloudNodelet::cameraInfoCallback, this);

        // No RGB in sync — colors come from class palette, not pixels.
        depth_sub_.subscribe(nh_,    "depth/image_rect_raw", queue_size_);
        semantic_sub_.subscribe(nh_, "semantic/image",       queue_size_);

        if (use_uncertainty_) {
            uncertainty_sub_.subscribe(nh_, "semantic/uncertainty", queue_size_);
            sync3_ = std::make_shared<Sync3>(SyncPolicy3(queue_size_),
                                             depth_sub_, semantic_sub_, uncertainty_sub_);
            sync3_->setMaxIntervalDuration(ros::Duration(0.5));
            sync3_->registerCallback(boost::bind(&SemanticCloudNodelet::syncCallback3,
                                                 this, _1, _2, _3));
            NODELET_INFO("  Sync: 3-topic (depth + semantic + uncertainty)");
        } else {
            sync2_ = std::make_shared<Sync2>(SyncPolicy2(queue_size_),
                                             depth_sub_, semantic_sub_);
            sync2_->setMaxIntervalDuration(ros::Duration(0.5));
            sync2_->registerCallback(boost::bind(&SemanticCloudNodelet::syncCallback2,
                                                 this, _1, _2));
            NODELET_INFO("  Sync: 2-topic (depth + semantic), uncertainty=%.2f",
                         default_uncertainty_);
        }
    }

    void setupPublishers() {
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_cloud", 1);
    }

    // =========================================================================
    // Callbacks
    // =========================================================================

    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        fx_ = msg->K[0]; fy_ = msg->K[4];
        cx_ = msg->K[2]; cy_ = msg->K[5];
        width_  = static_cast<int>(msg->width);
        height_ = static_cast<int>(msg->height);
        has_camera_info_ = true;
    }

    void syncCallback3(const sensor_msgs::Image::ConstPtr& depth_msg,
                       const sensor_msgs::Image::ConstPtr& semantic_msg,
                       const sensor_msgs::Image::ConstPtr& uncertainty_msg) {
        if (!has_camera_info_) {
            NODELET_WARN_THROTTLE(1.0, "[cloud] waiting for camera info");
            return;
        }
        tryConfigureCompactPalette();

        cv_bridge::CvImageConstPtr depth_cv, semantic_cv, uncertainty_cv;
        try {
            depth_cv       = cv_bridge::toCvShare(depth_msg);
            semantic_cv    = cv_bridge::toCvShare(semantic_msg);
            uncertainty_cv = cv_bridge::toCvShare(uncertainty_msg, "mono8");
        } catch (cv_bridge::Exception& e) {
            NODELET_ERROR("[cloud] cv_bridge: %s", e.what());
            return;
        }

        // Use PointCloud2Ptr for zero-copy publish within nodelet manager
        sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2);
        generateCloud(depth_cv->image, semantic_cv->image,
                      &uncertainty_cv->image, depth_msg->header, *cloud_msg);
        cloud_pub_.publish(cloud_msg);
    }

    void syncCallback2(const sensor_msgs::Image::ConstPtr& depth_msg,
                       const sensor_msgs::Image::ConstPtr& semantic_msg) {
        if (!has_camera_info_) {
            NODELET_WARN_THROTTLE(1.0, "[cloud] waiting for camera info");
            return;
        }
        tryConfigureCompactPalette();

        cv_bridge::CvImageConstPtr depth_cv, semantic_cv;
        try {
            depth_cv    = cv_bridge::toCvShare(depth_msg);
            semantic_cv = cv_bridge::toCvShare(semantic_msg);
        } catch (cv_bridge::Exception& e) {
            NODELET_ERROR("[cloud] cv_bridge: %s", e.what());
            return;
        }

        sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2);
        generateCloud(depth_cv->image, semantic_cv->image,
                      nullptr, depth_msg->header, *cloud_msg);
        cloud_pub_.publish(cloud_msg);
    }

    // =========================================================================
    // Point Cloud Generation (identical logic to semantic_cloud_node.cpp)
    // =========================================================================

    void generateCloud(const cv::Mat& depth,
                       const cv::Mat& semantic,
                       const cv::Mat* uncertainty,
                       const std_msgs::Header& header,
                       sensor_msgs::PointCloud2& cloud_msg) {

        const int out_w = width_  / downsample_factor_;
        const int out_h = height_ / downsample_factor_;

        cv::Mat depth_s, sem_s, unc_s;
        if (downsample_factor_ > 1) {
            cv::resize(depth,    depth_s, {out_w, out_h}, 0, 0, cv::INTER_NEAREST);
            cv::resize(semantic, sem_s,   {out_w, out_h}, 0, 0, cv::INTER_NEAREST);
            if (uncertainty)
                cv::resize(*uncertainty, unc_s, {out_w, out_h}, 0, 0, cv::INTER_LINEAR);
        } else {
            depth_s = depth; sem_s = semantic;
            if (uncertainty) unc_s = *uncertainty;
        }

        const float fx = static_cast<float>(fx_ / downsample_factor_);
        const float fy = static_cast<float>(fy_ / downsample_factor_);
        const float cx = static_cast<float>(cx_ / downsample_factor_);
        const float cy = static_cast<float>(cy_ / downsample_factor_);
        const bool has_unc = !unc_s.empty();
        const bool depth_is_16u = (depth_s.type() == CV_16UC1);

        // Fill PointCloud2 metadata
        cloud_msg.header = header;
        if (!output_frame_.empty()) cloud_msg.header.frame_id = output_frame_;
        cloud_msg.height      = 1;
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense    = true;
        cloud_msg.point_step  = static_cast<uint32_t>(sizeof(SemanticPointPacked));

        cloud_msg.fields.resize(6);
        {
            auto mkfield = [](const char* name, uint32_t offset, uint8_t dtype) {
                sensor_msgs::PointField f;
                f.name = name; f.offset = offset; f.datatype = dtype; f.count = 1;
                return f;
            };
            cloud_msg.fields[0] = mkfield("x",           offsetof(SemanticPointPacked, x),           sensor_msgs::PointField::FLOAT32);
            cloud_msg.fields[1] = mkfield("y",           offsetof(SemanticPointPacked, y),           sensor_msgs::PointField::FLOAT32);
            cloud_msg.fields[2] = mkfield("z",           offsetof(SemanticPointPacked, z),           sensor_msgs::PointField::FLOAT32);
            cloud_msg.fields[3] = mkfield("rgb",         offsetof(SemanticPointPacked, rgb),         sensor_msgs::PointField::UINT32);
            cloud_msg.fields[4] = mkfield("label",       offsetof(SemanticPointPacked, label),       sensor_msgs::PointField::UINT32);
            cloud_msg.fields[5] = mkfield("uncertainty", offsetof(SemanticPointPacked, uncertainty), sensor_msgs::PointField::FLOAT32);
        }

        // Allocate worst-case, write directly, shrink at end
        const int max_pts = out_w * out_h;
        cloud_msg.data.resize(static_cast<size_t>(max_pts) * sizeof(SemanticPointPacked));
        SemanticPointPacked* out_ptr = reinterpret_cast<SemanticPointPacked*>(cloud_msg.data.data());
        uint32_t count = 0;

        const uint32_t num_cls = static_cast<uint32_t>(num_classes_);

        for (int v = 0; v < out_h; ++v) {
            const uint8_t* sem_row = sem_s.ptr<uint8_t>(v);
            const uint8_t* unc_row = has_unc ? unc_s.ptr<uint8_t>(v) : nullptr;

            for (int u = 0; u < out_w; ++u) {
                float d;
                if (depth_is_16u)
                    d = static_cast<float>(depth_s.at<uint16_t>(v, u)) * 0.001f;
                else
                    d = depth_s.at<float>(v, u);

                if (d <= min_depth_ || d >= max_depth_ || !std::isfinite(d))
                    continue;

                const uint32_t label = sem_row[u];
                if (label >= num_cls) continue;

                SemanticPointPacked& p = out_ptr[count++];
                p.z = d;
                p.x = (static_cast<float>(u) - cx) * d / fx;
                p.y = (static_cast<float>(v) - cy) * d / fy;

                const auto& c = class_colors_[label];
                p.rgb = (static_cast<uint32_t>(c[0]) << 16) |
                        (static_cast<uint32_t>(c[1]) <<  8) |
                         static_cast<uint32_t>(c[2]);

                p.label = label;
                p.uncertainty = has_unc
                    ? static_cast<float>(unc_row[u]) * (1.0f / 255.0f)
                    : default_uncertainty_;
            }
        }

        cloud_msg.data.resize(static_cast<size_t>(count) * sizeof(SemanticPointPacked));
        cloud_msg.width    = count;
        cloud_msg.row_step = cloud_msg.point_step * count;
    }

    // =========================================================================
    // Members
    // =========================================================================

    ros::NodeHandle nh_, pnh_;

    ros::Subscriber camera_info_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_, semantic_sub_, uncertainty_sub_;

    // 3-topic sync: depth + semantic + uncertainty
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image,
        sensor_msgs::Image> SyncPolicy3;
    typedef message_filters::Synchronizer<SyncPolicy3> Sync3;
    std::shared_ptr<Sync3> sync3_;

    // 2-topic sync: depth + semantic (no uncertainty)
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image> SyncPolicy2;
    typedef message_filters::Synchronizer<SyncPolicy2> Sync2;
    std::shared_ptr<Sync2> sync2_;

    ros::Publisher cloud_pub_;

    // Camera
    double fx_ = 386.0, fy_ = 386.0, cx_ = 320.0, cy_ = 240.0;
    int width_ = 640, height_ = 480;
    bool has_camera_info_ = false;

    // Dataset / classes
    std::string dataset_;
    std::vector<std::array<uint8_t, 3>> class_colors_;
    std::vector<std::array<uint8_t, 3>> base_class_colors_;
    std::vector<std::array<uint8_t, 3>> compact_colors_;
    std::vector<int> relevant_classes_;
    bool compact_palette_configured_ = false;
    int num_classes_ = 37;

    // Processing
    int         downsample_factor_   = 2;
    double      min_depth_           = 0.06;
    double      max_depth_           = 6.0;
    int         queue_size_          = 10;
    bool        use_uncertainty_     = true;
    float       default_uncertainty_ = 0.3f;
    std::string output_frame_;
};

} // namespace hesfm

PLUGINLIB_EXPORT_CLASS(hesfm::SemanticCloudNodelet, nodelet::Nodelet)
