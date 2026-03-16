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
        
        ROS_INFO("Semantic Cloud Node initialized");
        ROS_INFO("  Downsample factor: %d", downsample_factor_);
        ROS_INFO("  Min depth: %.2f m, Max depth: %.2f m", min_depth_, max_depth_);
    }

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        // Processing parameters
        pnh_.param("downsample_factor", downsample_factor_, 2);
        pnh_.param("min_depth", min_depth_, 0.1);
        pnh_.param("max_depth", max_depth_, 6.0);
        pnh_.param("num_classes", num_classes_, 40);
        
        // Default uncertainty if not provided
        pnh_.param("default_uncertainty", default_uncertainty_, 0.3);
        
        // Queue sizes
        pnh_.param("queue_size", queue_size_, 10);
        
        // Frame ID override
        pnh_.param<std::string>("output_frame", output_frame_, "");
    }
    
    void setupSubscribers() {
        // Camera info (non-synchronized, just store latest)
        camera_info_sub_ = nh_.subscribe("color/camera_info", 1,
                                          &SemanticCloudNode::cameraInfoCallback, this);
        
        // Synchronized subscribers for RGB, depth, and semantic
        rgb_sub_.subscribe(nh_, "color/image_raw", queue_size_);
        depth_sub_.subscribe(nh_, "depth/image_rect_raw", queue_size_);
        semantic_sub_.subscribe(nh_, "semantic/image", queue_size_);
        
        // Approximate time synchronizer
        sync_ = std::make_shared<Sync>(SyncPolicy(queue_size_),
                                        rgb_sub_, depth_sub_, semantic_sub_);
        sync_->registerCallback(boost::bind(&SemanticCloudNode::syncCallback,
                                            this, _1, _2, _3));
        
        ROS_INFO("Subscribed to synchronized topics");
    }
    
    void setupPublishers() {
        // Semantic point cloud publisher
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("semantic_cloud", 1);
        
        // Debug visualization (optional)
        debug_pub_ = it_.advertise("semantic_cloud/debug", 1);
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
                      const sensor_msgs::Image::ConstPtr& semantic_msg) {
        
        if (!has_camera_info_) {
            ROS_WARN_THROTTLE(1.0, "Waiting for camera info...");
            return;
        }
        
        ros::Time start_time = ros::Time::now();
        
        // Convert images
        cv_bridge::CvImageConstPtr rgb_cv, depth_cv, semantic_cv;
        
        try {
            rgb_cv = cv_bridge::toCvShare(rgb_msg, "bgr8");
            depth_cv = cv_bridge::toCvShare(depth_msg);
            semantic_cv = cv_bridge::toCvShare(semantic_msg);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        
        // Generate semantic point cloud
        sensor_msgs::PointCloud2 cloud_msg;
        generateSemanticCloud(rgb_cv->image, depth_cv->image, semantic_cv->image,
                              rgb_msg->header, cloud_msg);
        
        // Publish
        cloud_pub_.publish(cloud_msg);
        
        // Debug timing
        double processing_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        ROS_DEBUG("Generated semantic cloud with %d points in %.1f ms",
                  cloud_msg.width * cloud_msg.height, processing_time);
    }
    
    // =========================================================================
    // Point Cloud Generation
    // =========================================================================
    
    void generateSemanticCloud(const cv::Mat& rgb,
                                const cv::Mat& depth,
                                const cv::Mat& semantic,
                                const std_msgs::Header& header,
                                sensor_msgs::PointCloud2& cloud_msg) {
        
        // Determine output dimensions
        int out_width = width_ / downsample_factor_;
        int out_height = height_ / downsample_factor_;
        
        // Resize images if needed
        cv::Mat depth_resized, semantic_resized, rgb_resized;
        
        if (downsample_factor_ > 1) {
            cv::resize(depth, depth_resized, cv::Size(out_width, out_height), 
                       0, 0, cv::INTER_NEAREST);
            cv::resize(semantic, semantic_resized, cv::Size(out_width, out_height),
                       0, 0, cv::INTER_NEAREST);
            cv::resize(rgb, rgb_resized, cv::Size(out_width, out_height),
                       0, 0, cv::INTER_LINEAR);
        } else {
            depth_resized = depth;
            semantic_resized = semantic;
            rgb_resized = rgb;
        }
        
        // Adjust intrinsics for downsampling
        double fx = fx_ / downsample_factor_;
        double fy = fy_ / downsample_factor_;
        double cx = cx_ / downsample_factor_;
        double cy = cy_ / downsample_factor_;
        
        // Count valid points
        int valid_count = 0;
        for (int v = 0; v < out_height; ++v) {
            for (int u = 0; u < out_width; ++u) {
                float d = getDepthValue(depth_resized, u, v);
                if (d > min_depth_ && d < max_depth_ && std::isfinite(d)) {
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
                
                if (d <= min_depth_ || d >= max_depth_ || !std::isfinite(d)) {
                    continue;
                }
                
                // Back-project to 3D
                float x = (u - cx) * d / fx;
                float y = (v - cy) * d / fy;
                float z = d;
                
                // Get semantic label
                uint32_t label = getSemanticLabel(semantic_resized, u, v);
                
                // Get RGB
                cv::Vec3b bgr = rgb_resized.at<cv::Vec3b>(v, u);
                uint32_t rgb = (static_cast<uint32_t>(bgr[2]) << 16) |
                               (static_cast<uint32_t>(bgr[1]) << 8) |
                               static_cast<uint32_t>(bgr[0]);
                
                // Compute depth-based uncertainty
                float uncertainty = computeUncertainty(d, u, v, out_width, out_height);
                
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
    
    float computeUncertainty(float depth, int u, int v, int width, int height) {
        // Base uncertainty from depth (increases with distance)
        float depth_uncertainty = 0.1f + 0.1f * (depth / max_depth_);
        
        // Edge uncertainty (higher near image borders)
        float edge_x = std::min(u, width - u) / static_cast<float>(width / 2);
        float edge_y = std::min(v, height - v) / static_cast<float>(height / 2);
        float edge_uncertainty = 0.1f * (1.0f - std::min(edge_x, edge_y));
        
        return std::min(1.0f, depth_uncertainty + edge_uncertainty);
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
    
    // Synchronizer
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;
    
    // Publishers
    ros::Publisher cloud_pub_;
    image_transport::Publisher debug_pub_;
    
    // Camera parameters
    double fx_ = 386.0, fy_ = 386.0;
    double cx_ = 320.0, cy_ = 240.0;
    int width_ = 640, height_ = 480;
    bool has_camera_info_ = false;
    
    // Processing parameters
    int downsample_factor_;
    double min_depth_;
    double max_depth_;
    int num_classes_;
    double default_uncertainty_;
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
