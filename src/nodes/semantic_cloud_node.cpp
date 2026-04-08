/**
 * @file semantic_cloud_node.cpp
 * @brief ROS node to generate semantic point clouds from RGB-D and segmentation
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node combines depth images with semantic segmentation to produce
 * semantic point clouds for the HESFM mapper.
 * 
 * Subscriptions:
 *   - color/image_raw (sensor_msgs/Image): RGB image
 *   - depth/image_rect_raw (sensor_msgs/Image): Depth image
 *   - color/camera_info (sensor_msgs/CameraInfo): Camera intrinsics
 *   - semantic/image (sensor_msgs/Image): Semantic segmentation labels
 *   - semantic/probabilities (sensor_msgs/Image): Class probabilities (optional)
 * 
 * Publications:
 *   - semantic_cloud (sensor_msgs/PointCloud2): Semantic point cloud
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "hesfm/types.h"

/**
 * @brief Semantic Point Cloud Generator Node
 */
class SemanticCloudNode {
public:
    SemanticCloudNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh), it_(nh) {
        
        loadParameters();
        setupSubscribers();
        setupPublishers();
        
        ROS_IN O("Semantic Cloud Node initialized");
        ROS_INFO("  Dataset:          %s (%d classes)", dataset_.c_str(), num_classes_);
        ROS_INFO("  Downsample factor: %d", downsample_factor_);
        ROS_INFO("  Min depth: %.2f m, Max depth: %.2f m", min_depth_, max_depth_);
        ROS_INFO("  Output frame: %s", output_frame_.empty() ? "camera frame" : output_frame_.c_str());
    }

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        // Dataset selection: 'nyuv2' (40 classes) or 'sunrgbd' (37 classes)
        pnh_.param<std::string>("esanet_dataset", dataset_, "nyuv2");
        if (dataset_ == "sunrgbd") {
            num_classes_       = hesfm::SUNRGBD_NUM_CLASSES;
            class_names_       = hesfm::SUNRGBD_CLASS_NAMES;
            traversable_set_   = hesfm::SUNRGBD_TRAVERSABLE_CLASSES;
            class_colors_      = hesfm::SUNRGBD_CLASS_COLORS;
        } else {
            num_classes_       = hesfm::NYUV2_NUM_CLASSES;
            class_names_       = hesfm::NYUV2_CLASS_NAMES;
            traversable_set_   = hesfm::DEFAULT_TRAVERSABLE_CLASSES;
            class_colors_      = hesfm::NYUV2_CLASS_COLORS;
        }

        // Allow explicit override from launch file
        pnh_.param("num_classes", num_classes_, num_classes_);

        // Processing parameters
        pnh_.param("downsample_factor", downsample_factor_, 2);
        pnh_.param("min_depth", min_depth_, 0.06);
        pnh_.param("max_depth", max_depth_, 6.0);

        // Queue sizes
        pnh_.param("queue_size", queue_size_, 10);

        // Frame ID override
        pnh_.param<std::string>("output_frame", output_frame_, "");
    }
    
    void setupSubscribers() {
        // Camera info (non-synchronized, just store latest)
        camera_info_sub_ = nh_.subscribe("color/camera_info", 1,
                                          &SemanticCloudNode::cameraInfoCallback, this);

        // Synchronized subscribers for RGB, depth, semantic, and uncertainty
        rgb_sub_.subscribe(nh_, "color/image_raw", queue_size_);
        depth_sub_.subscribe(nh_, "depth/image_rect_raw", queue_size_);
        semantic_sub_.subscribe(nh_, "semantic/image", queue_size_);
        uncertainty_sub_.subscribe(nh_, "semantic/uncertainty", queue_size_);

        // Approximate time synchronizer (4-topic)
        sync_ = std::make_shared<Sync>(SyncPolicy(queue_size_),
                                        rgb_sub_, depth_sub_, semantic_sub_,
                                        uncertainty_sub_);
        sync_->setMaxIntervalDuration(ros::Duration(0.5));
        sync_->registerCallback(boost::bind(&SemanticCloudNode::syncCallback,
                                            this, _1, _2, _3, _4));
    }
    
    void setupPublishers() {
        // Semantic point cloud publisher
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_cloud", 1);

        // Debug visualization (optional)
        // debug_pub_ = it_.advertise("semantic_cloud/debug", 1);
    }
    
    // =========================================================================
    // Callbacks
    // =========================================================================
    
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        // Store camera intrinsics
        fx_ = msg->K[0];
        fy_ = msg->K[4];
        cx_ = msg->K[2];
        cy_ = msg->K[5];
        
        width_ = msg->width;
        height_ = msg->height;
        
        has_camera_info_ = true;
    }
    
    void syncCallback(const sensor_msgs::Image::ConstPtr& rgb_msg,
                      const sensor_msgs::Image::ConstPtr& depth_msg,
                      const sensor_msgs::Image::ConstPtr& semantic_msg,
                      const sensor_msgs::Image::ConstPtr& uncertainty_msg) {

        if (!has_camera_info_) {
            ROS_WARN_THROTTLE(1.0, "[cloud] waiting for camera info");
            return;
        }

        cv_bridge::CvImageConstPtr rgb_cv, depth_cv, semantic_cv, uncertainty_cv;
        try {
            rgb_cv         = cv_bridge::toCvShare(rgb_msg, "bgr8");
            depth_cv       = cv_bridge::toCvShare(depth_msg);
            semantic_cv    = cv_bridge::toCvShare(semantic_msg);
            uncertainty_cv = cv_bridge::toCvShare(uncertainty_msg, "mono8");
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("[cloud] cv_bridge: %s", e.what());
            return;
        }

        sensor_msgs::PointCloud2 cloud_msg;
        generateSemanticCloud(rgb_cv->image, depth_cv->image, semantic_cv->image,
                              uncertainty_cv->image, rgb_msg->header, cloud_msg);

        cloud_pub_.publish(cloud_msg);
    }
    
    // =========================================================================
    // Point Cloud Generation
    // =========================================================================
    
    void generateSemanticCloud(const cv::Mat& rgb,
                                const cv::Mat& depth,
                                const cv::Mat& semantic,
                                const cv::Mat& uncertainty,
                                const std_msgs::Header& header,
                                sensor_msgs::PointCloud2& cloud_msg) {

        // Determine output dimensions
        int out_width = width_ / downsample_factor_;
        int out_height = height_ / downsample_factor_;

        // Resize images if needed
        cv::Mat depth_resized, semantic_resized, rgb_resized, uncertainty_resized;

        if (downsample_factor_ > 1) {
            cv::resize(depth, depth_resized, cv::Size(out_width, out_height),
                       0, 0, cv::INTER_NEAREST);
            cv::resize(semantic, semantic_resized, cv::Size(out_width, out_height),
                       0, 0, cv::INTER_NEAREST);
            cv::resize(rgb, rgb_resized, cv::Size(out_width, out_height),
                       0, 0, cv::INTER_LINEAR);
            cv::resize(uncertainty, uncertainty_resized, cv::Size(out_width, out_height),
                       0, 0, cv::INTER_LINEAR);
        } else {
            depth_resized = depth;
            semantic_resized = semantic;
            rgb_resized = rgb;
            uncertainty_resized = uncertainty;
        }
        
        // Adjust intrinsics for downsampling
        double fx = fx_ / downsample_factor_;
        double fy = fy_ / downsample_factor_;
        double cx = cx_ / downsample_factor_;
        double cy = cy_ / downsample_factor_;
        
        // Count valid points (valid depth AND valid semantic label)
        int valid_count = 0;
        for (int v = 0; v < out_height; ++v) {
            for (int u = 0; u < out_width; ++u) {
                float d = getDepthValue(depth_resized, u, v);
                uint32_t label = getSemanticLabel(semantic_resized, u, v);
                if (d > min_depth_ && d < max_depth_ && std::isfinite(d) &&
                    label < static_cast<uint32_t>(num_classes_)) {
                    valid_count++;
                }
            }
        }
        
        // Setup PointCloud2 fields
        cloud_msg.header = header;
        if (!output_frame_.empty()) {
            cloud_msg.header.frame_id = output_frame_;
        }
        
        cloud_msg.height = 1;
        cloud_msg.width = valid_count;
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense = true;
        
        // Define fields: x, y, z, rgb, label, uncertainty
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2Fields(6,
            "x", 1, sensor_msgs::PointField::FLOAT32,
            "y", 1, sensor_msgs::PointField::FLOAT32,
            "z", 1, sensor_msgs::PointField::FLOAT32,
            "rgb", 1, sensor_msgs::PointField::UINT32,
            "label", 1, sensor_msgs::PointField::UINT32,
            "uncertainty", 1, sensor_msgs::PointField::FLOAT32);
        
        modifier.resize(valid_count);
        
        // Fill point cloud
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgb(cloud_msg, "rgb");
        sensor_msgs::PointCloud2Iterator<uint32_t> iter_label(cloud_msg, "label");
        sensor_msgs::PointCloud2Iterator<float> iter_unc(cloud_msg, "uncertainty");
        
        for (int v = 0; v < out_height; ++v) {
            for (int u = 0; u < out_width; ++u) {
                float d = getDepthValue(depth_resized, u, v);
                
                // Get semantic label early — skip void (255) and out-of-range
                uint32_t label = getSemanticLabel(semantic_resized, u, v);
                if (d <= min_depth_ || d >= max_depth_ || !std::isfinite(d) ||
                    label >= static_cast<uint32_t>(num_classes_)) {
                    continue;
                }
                
                // Back-project to 3D
                float x = (u - cx) * d / fx;
                float y = (v - cy) * d / fy;
                float z = d;
                
                // Color by semantic class
                const auto& color = class_colors_[label];
                uint32_t rgb = (static_cast<uint32_t>(color[0]) << 16) |
                               (static_cast<uint32_t>(color[1]) << 8) |
                               static_cast<uint32_t>(color[2]);
                
                // Use real per-pixel semantic uncertainty from segmentation node
                // (published as mono8: 0-255 → 0.0-1.0)
                float uncertainty = static_cast<float>(
                    uncertainty_resized.at<uint8_t>(v, u)) / 255.0f;
                
                // Write to point cloud
                *iter_x = x;
                *iter_y = y;
                *iter_z = z;
                *iter_rgb = rgb;
                *iter_label = label;
                *iter_unc = uncertainty;
                
                ++iter_x; ++iter_y; ++iter_z;
                ++iter_rgb; ++iter_label; ++iter_unc;
            }
        }
    }
    
    float getDepthValue(const cv::Mat& depth, int u, int v) {
        switch (depth.type()) {
            case CV_16UC1: {
                uint16_t d = depth.at<uint16_t>(v, u);
                return static_cast<float>(d) * 0.001f;  // mm to m
            }
            case CV_32FC1:
                return depth.at<float>(v, u);
            default:
                return 0.0f;
        }
    }
    
    uint32_t getSemanticLabel(const cv::Mat& semantic, int u, int v) {
        switch (semantic.type()) {
            case CV_8UC1:
                return static_cast<uint32_t>(semantic.at<uint8_t>(v, u));
            case CV_16UC1:
                return static_cast<uint32_t>(semantic.at<uint16_t>(v, u));
            case CV_32SC1:
                return static_cast<uint32_t>(semantic.at<int32_t>(v, u));
            default:
                return 0;
        }
    }
    
    // =========================================================================
    // Member Variables
    // =========================================================================
    
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::ImageTransport it_;
    
    // Subscribers
    ros::Subscriber camera_info_sub_;
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> semantic_sub_;
    message_filters::Subscriber<sensor_msgs::Image> uncertainty_sub_;

    // Synchronizer (4 topics: rgb, depth, semantic labels, semantic uncertainty)
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;
    
    // Publishers
    ros::Publisher cloud_pub_;
    // image_transport::Publisher debug_pub_;  // uncomment for debug visualization
    
    // Camera parameters
    double fx_ = 386.0, fy_ = 386.0;
    double cx_ = 320.0, cy_ = 240.0;
    int width_ = 640, height_ = 480;
    bool has_camera_info_ = false;
    
    // Dataset
    std::string dataset_;
    std::vector<std::string> class_names_;
    std::set<int> traversable_set_;
    std::vector<std::array<uint8_t,3>> class_colors_;

    // Processing parameters
    int downsample_factor_;
    double min_depth_;
    double max_depth_;
    int num_classes_;
    int queue_size_;
    std::string output_frame_;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "semantic_cloud_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    try {
        SemanticCloudNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    
    return 0;
}
