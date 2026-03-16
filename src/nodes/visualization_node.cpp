/**
 * @file visualization_node.cpp
 * @brief ROS node for visualizing HESFM semantic maps
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 * 
 * This node provides rich visualization of semantic maps including
 * colored point clouds, uncertainty overlays, and class-specific views.
 * 
 * Subscriptions:
 *   - semantic_map (sensor_msgs/PointCloud2): Semantic map from mapper
 *   - primitives (visualization_msgs/MarkerArray): Gaussian primitives
 * 
 * Publications:
 *   - visualization/semantic (sensor_msgs/PointCloud2): Colored by class
 *   - visualization/uncertainty (sensor_msgs/PointCloud2): Colored by uncertainty
 *   - visualization/confidence (sensor_msgs/PointCloud2): Colored by confidence
 *   - visualization/class_X (sensor_msgs/PointCloud2): Single class view
 *   - visualization/legend (visualization_msgs/MarkerArray): Class legend
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>
#include <dynamic_reconfigure/server.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

#include "hesfm/types.h"

/**
 * @brief Color palette for semantic classes (NYUv2 40-class)
 */
struct ClassColorPalette {
    struct RGB {
        uint8_t r, g, b;
    };
    
    static const std::vector<RGB>& getColors() {
        static std::vector<RGB> colors = {
            {128, 128, 128},  // 0: wall - gray
            {139, 119, 101},  // 1: floor - brown
            {244, 164, 96},   // 2: cabinet - sandy brown
            {255, 182, 193},  // 3: bed - light pink
            {255, 215, 0},    // 4: chair - gold
            {220, 20, 60},    // 5: sofa - crimson
            {255, 140, 0},    // 6: table - dark orange
            {139, 69, 19},    // 7: door - saddle brown
            {135, 206, 235},  // 8: window - sky blue
            {160, 82, 45},    // 9: bookshelf - sienna
            {255, 105, 180},  // 10: picture - hot pink
            {0, 128, 128},    // 11: counter - teal
            {210, 180, 140},  // 12: blinds - tan
            {70, 130, 180},   // 13: desk - steel blue
            {188, 143, 143},  // 14: shelves - rosy brown
            {147, 112, 219},  // 15: curtain - medium purple
            {222, 184, 135},  // 16: dresser - burlywood
            {255, 228, 225},  // 17: pillow - misty rose
            {192, 192, 192},  // 18: mirror - silver
            {139, 119, 101},  // 19: floor_mat - brown
            {128, 0, 128},    // 20: clothes - purple
            {245, 245, 245},  // 21: ceiling - white smoke
            {139, 90, 43},    // 22: books - peru
            {173, 216, 230},  // 23: fridge - light blue
            {0, 0, 139},      // 24: television - dark blue
            {255, 255, 224},  // 25: paper - light yellow
            {240, 255, 255},  // 26: towel - azure
            {176, 224, 230},  // 27: shower_curtain - powder blue
            {210, 105, 30},   // 28: box - chocolate
            {255, 255, 255},  // 29: whiteboard - white
            {255, 0, 0},      // 30: person - red
            {85, 107, 47},    // 31: night_stand - dark olive green
            {255, 255, 240},  // 32: toilet - ivory
            {176, 196, 222},  // 33: sink - light steel blue
            {255, 250, 205},  // 34: lamp - lemon chiffon
            {224, 255, 255},  // 35: bathtub - light cyan
            {75, 0, 130},     // 36: bag - indigo
            {169, 169, 169},  // 37: otherstructure - dark gray
            {105, 105, 105},  // 38: otherfurniture - dim gray
            {128, 128, 0},    // 39: otherprop - olive
        };
        return colors;
    }
    
    static RGB getColor(int class_id) {
        const auto& colors = getColors();
        if (class_id >= 0 && class_id < static_cast<int>(colors.size())) {
            return colors[class_id];
        }
        return {128, 128, 128};  // Default gray
    }
};

/**
 * @brief Visualization Node
 */
class VisualizationNode {
public:
    VisualizationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh) {
        
        loadParameters();
        setupSubscribers();
        setupPublishers();
        
        ROS_INFO("Visualization Node initialized");
    }

private:
    // =========================================================================
    // Initialization
    // =========================================================================
    
    void loadParameters() {
        pnh_.param("num_classes", num_classes_, 40);
        pnh_.param("point_size", point_size_, 0.05);
        pnh_.param("publish_legend", publish_legend_, true);
        pnh_.param("highlight_class", highlight_class_, -1);
        
        // Get class names
        class_names_ = hesfm::NYUV2_CLASS_NAMES;
    }
    
    void setupSubscribers() {
        // Semantic map subscriber
        map_sub_ = nh_.subscribe("semantic_map", 1,
                                  &VisualizationNode::mapCallback, this);
    }
    
    void setupPublishers() {
        // Colored by semantic class
        semantic_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
            "visualization/semantic", 1);
        
        // Colored by uncertainty
        uncertainty_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
            "visualization/uncertainty", 1);
        
        // Colored by confidence
        confidence_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
            "visualization/confidence", 1);
        
        // Class legend
        legend_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "visualization/legend", 1, true);
        
        // Publish legend once
        if (publish_legend_) {
            publishLegend();
        }
    }
    
    // =========================================================================
    // Callbacks
    // =========================================================================
    
    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        // Parse input point cloud
        std::vector<MapPoint> points;
        parsePointCloud(msg, points);
        
        if (points.empty()) return;
        
        // Publish different visualizations
        publishSemanticCloud(msg->header, points);
        publishUncertaintyCloud(msg->header, points);
        publishConfidenceCloud(msg->header, points);
    }
    
    // =========================================================================
    // Point Cloud Parsing
    // =========================================================================
    
    struct MapPoint {
        float x, y, z;
        int label;
        float confidence;
        float uncertainty;
    };
    
    void parsePointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg,
                          std::vector<MapPoint>& points) {
        
        // Find field indices
        int x_idx = -1, y_idx = -1, z_idx = -1;
        int label_idx = -1, conf_idx = -1, unc_idx = -1;
        int rgb_idx = -1;
        
        for (size_t i = 0; i < msg->fields.size(); ++i) {
            const auto& field = msg->fields[i];
            if (field.name == "x") x_idx = i;
            else if (field.name == "y") y_idx = i;
            else if (field.name == "z") z_idx = i;
            else if (field.name == "label") label_idx = i;
            else if (field.name == "confidence") conf_idx = i;
            else if (field.name == "uncertainty") unc_idx = i;
            else if (field.name == "rgb") rgb_idx = i;
        }
        
        if (x_idx < 0 || y_idx < 0 || z_idx < 0) return;
        
        points.reserve(msg->width * msg->height);
        
        for (size_t i = 0; i < msg->width * msg->height; ++i) {
            const uint8_t* ptr = &msg->data[i * msg->point_step];
            
            MapPoint pt;
            memcpy(&pt.x, ptr + msg->fields[x_idx].offset, sizeof(float));
            memcpy(&pt.y, ptr + msg->fields[y_idx].offset, sizeof(float));
            memcpy(&pt.z, ptr + msg->fields[z_idx].offset, sizeof(float));
            
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                continue;
            }
            
            // Get label - infer from RGB if not directly available
            if (label_idx >= 0) {
                uint32_t label;
                memcpy(&label, ptr + msg->fields[label_idx].offset, sizeof(uint32_t));
                pt.label = static_cast<int>(label);
            } else if (rgb_idx >= 0) {
                // Infer class from RGB color
                uint32_t rgb;
                memcpy(&rgb, ptr + msg->fields[rgb_idx].offset, sizeof(uint32_t));
                uint8_t r = (rgb >> 16) & 0xFF;
                uint8_t g = (rgb >> 8) & 0xFF;
                uint8_t b = rgb & 0xFF;
                pt.label = inferClassFromColor(r, g, b);
            } else {
                pt.label = 0;
            }
            
            // Get confidence/uncertainty
            if (conf_idx >= 0) {
                memcpy(&pt.confidence, ptr + msg->fields[conf_idx].offset, sizeof(float));
            } else {
                pt.confidence = 0.8f;
            }
            
            if (unc_idx >= 0) {
                memcpy(&pt.uncertainty, ptr + msg->fields[unc_idx].offset, sizeof(float));
            } else {
                pt.uncertainty = 1.0f - pt.confidence;
            }
            
            points.push_back(pt);
        }
    }
    
    int inferClassFromColor(uint8_t r, uint8_t g, uint8_t b) {
        // Find closest matching class color
        const auto& colors = ClassColorPalette::getColors();
        int best_class = 0;
        int min_dist = std::numeric_limits<int>::max();
        
        for (size_t i = 0; i < colors.size(); ++i) {
            int dr = static_cast<int>(r) - static_cast<int>(colors[i].r);
            int dg = static_cast<int>(g) - static_cast<int>(colors[i].g);
            int db = static_cast<int>(b) - static_cast<int>(colors[i].b);
            int dist = dr*dr + dg*dg + db*db;
            
            if (dist < min_dist) {
                min_dist = dist;
                best_class = static_cast<int>(i);
            }
        }
        
        return best_class;
    }
    
    // =========================================================================
    // Publishing
    // =========================================================================
    
    void publishSemanticCloud(const std_msgs::Header& header,
                               const std::vector<MapPoint>& points) {
        
        if (semantic_pub_.getNumSubscribers() == 0) return;
        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.reserve(points.size());
        
        for (const auto& pt : points) {
            // Skip if highlighting specific class
            if (highlight_class_ >= 0 && pt.label != highlight_class_) {
                continue;
            }
            
            pcl::PointXYZRGB p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            
            auto color = ClassColorPalette::getColor(pt.label);
            p.r = color.r;
            p.g = color.g;
            p.b = color.b;
            
            cloud.push_back(p);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header = header;
        
        semantic_pub_.publish(msg);
    }
    
    void publishUncertaintyCloud(const std_msgs::Header& header,
                                  const std::vector<MapPoint>& points) {
        
        if (uncertainty_pub_.getNumSubscribers() == 0) return;
        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.reserve(points.size());
        
        for (const auto& pt : points) {
            pcl::PointXYZRGB p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            
            // Color map: low uncertainty (green) to high uncertainty (red)
            float u = std::clamp(pt.uncertainty, 0.0f, 1.0f);
            p.r = static_cast<uint8_t>(255 * u);
            p.g = static_cast<uint8_t>(255 * (1.0f - u));
            p.b = 0;
            
            cloud.push_back(p);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header = header;
        
        uncertainty_pub_.publish(msg);
    }
    
    void publishConfidenceCloud(const std_msgs::Header& header,
                                 const std::vector<MapPoint>& points) {
        
        if (confidence_pub_.getNumSubscribers() == 0) return;
        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.reserve(points.size());
        
        for (const auto& pt : points) {
            pcl::PointXYZRGB p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            
            // Color map: low confidence (blue) to high confidence (yellow)
            float c = std::clamp(pt.confidence, 0.0f, 1.0f);
            p.r = static_cast<uint8_t>(255 * c);
            p.g = static_cast<uint8_t>(255 * c);
            p.b = static_cast<uint8_t>(255 * (1.0f - c));
            
            cloud.push_back(p);
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header = header;
        
        confidence_pub_.publish(msg);
    }
    
    void publishLegend() {
        visualization_msgs::MarkerArray markers;
        
        const auto& colors = ClassColorPalette::getColors();
        
        for (size_t i = 0; i < class_names_.size() && i < colors.size(); ++i) {
            // Color cube
            visualization_msgs::Marker cube;
            cube.header.frame_id = "map";
            cube.header.stamp = ros::Time::now();
            cube.ns = "legend_colors";
            cube.id = static_cast<int>(i);
            cube.type = visualization_msgs::Marker::CUBE;
            cube.action = visualization_msgs::Marker::ADD;
            
            cube.pose.position.x = -5.0;
            cube.pose.position.y = 5.0 - static_cast<double>(i) * 0.3;
            cube.pose.position.z = 0.0;
            cube.pose.orientation.w = 1.0;
            
            cube.scale.x = cube.scale.y = cube.scale.z = 0.2;
            
            cube.color.r = colors[i].r / 255.0f;
            cube.color.g = colors[i].g / 255.0f;
            cube.color.b = colors[i].b / 255.0f;
            cube.color.a = 1.0f;
            
            cube.lifetime = ros::Duration(0);  // Forever
            
            markers.markers.push_back(cube);
            
            // Class name text
            visualization_msgs::Marker text;
            text.header = cube.header;
            text.ns = "legend_text";
            text.id = static_cast<int>(i);
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD;
            
            text.pose.position.x = -4.6;
            text.pose.position.y = cube.pose.position.y;
            text.pose.position.z = 0.0;
            
            text.scale.z = 0.15;
            text.color.r = text.color.g = text.color.b = 1.0;
            text.color.a = 1.0;
            
            char buf[64];
            snprintf(buf, sizeof(buf), "%zu: %s", i, class_names_[i].c_str());
            text.text = buf;
            
            text.lifetime = ros::Duration(0);
            
            markers.markers.push_back(text);
        }
        
        legend_pub_.publish(markers);
    }
    
    // =========================================================================
    // Member Variables
    // =========================================================================
    
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // Subscribers
    ros::Subscriber map_sub_;
    
    // Publishers
    ros::Publisher semantic_pub_;
    ros::Publisher uncertainty_pub_;
    ros::Publisher confidence_pub_;
    ros::Publisher legend_pub_;
    
    // Parameters
    int num_classes_;
    double point_size_;
    bool publish_legend_;
    int highlight_class_;
    std::vector<std::string> class_names_;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "visualization_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    try {
        VisualizationNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    
    return 0;
}
