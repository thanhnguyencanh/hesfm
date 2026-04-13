/**
 * @file test_primitives.cpp
 * @brief Unit tests for HESFM Gaussian primitives
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include "hesfm/gaussian_primitives.h"
#include "hesfm/types.h"

using namespace hesfm;

class PrimitivesTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.target_primitives = 32;
        config_.min_points_per_primitive = 5;
        config_.max_points_per_primitive = 100;
        config_.conflict_threshold = 0.3;
        config_.regularization = 0.001;
        config_.kmeans_max_iter = 50;
        
        builder_ = std::make_unique<GaussianPrimitiveBuilder>(config_);
    }
    
    std::vector<SemanticPoint> createCluster(
        const Eigen::Vector3d& center, 
        double spread, 
        int num_points,
        int semantic_class) {
        
        std::vector<SemanticPoint> points;
        for (int i = 0; i < num_points; ++i) {
            SemanticPoint p;
            p.position = center + Eigen::Vector3d(
                spread * (static_cast<double>(rand()) / RAND_MAX - 0.5),
                spread * (static_cast<double>(rand()) / RAND_MAX - 0.5),
                spread * (static_cast<double>(rand()) / RAND_MAX - 0.5));
            p.semantic_class = semantic_class;
            p.semantic_confidence = 0.8f;
            p.uncertainty_semantic = 0.2;
            p.uncertainty_total = 0.2;
            points.push_back(p);
        }
        return points;
    }
    
    PrimitiveConfig config_;
    std::unique_ptr<GaussianPrimitiveBuilder> builder_;
};

TEST_F(PrimitivesTest, SingleCluster) {
    auto points = createCluster(Eigen::Vector3d(1, 2, 0.5), 0.3, 50, 5);
    
    auto primitives = builder_->buildPrimitives(points);
    
    ASSERT_GE(primitives.size(), 1u);
    
    // Check centroid is near cluster center
    const auto& prim = primitives[0];
    EXPECT_NEAR(prim.centroid.x(), 1.0, 0.5);
    EXPECT_NEAR(prim.centroid.y(), 2.0, 0.5);
    EXPECT_NEAR(prim.centroid.z(), 0.5, 0.3);
    
    // Check semantic class
    EXPECT_EQ(prim.semantic_class, 5);
}

TEST_F(PrimitivesTest, MultipleClusters) {
    std::vector<SemanticPoint> points;
    
    // Create 3 distinct clusters
    auto c1 = createCluster(Eigen::Vector3d(0, 0, 0), 0.2, 30, 1);
    auto c2 = createCluster(Eigen::Vector3d(5, 0, 0), 0.2, 30, 5);
    auto c3 = createCluster(Eigen::Vector3d(0, 5, 0), 0.2, 30, 10);
    
    points.insert(points.end(), c1.begin(), c1.end());
    points.insert(points.end(), c2.begin(), c2.end());
    points.insert(points.end(), c3.begin(), c3.end());
    
    auto primitives = builder_->buildPrimitives(points);
    
    // Should have at least 3 primitives
    EXPECT_GE(primitives.size(), 3u);
}

TEST_F(PrimitivesTest, DSTFusion) {
    // Create two overlapping clusters with different classes
    auto c1 = createCluster(Eigen::Vector3d(0, 0, 0), 0.3, 30, 1);
    auto c2 = createCluster(Eigen::Vector3d(0.2, 0, 0), 0.3, 30, 5);
    
    std::vector<SemanticPoint> points;
    points.insert(points.end(), c1.begin(), c1.end());
    points.insert(points.end(), c2.begin(), c2.end());
    
    auto primitives = builder_->buildPrimitives(points);
    
    // Should still produce valid primitives
    EXPECT_GT(primitives.size(), 0u);
    
    // Check DST conflict is computed
    for (const auto& prim : primitives) {
        EXPECT_GE(prim.conflict, 0.0);
        EXPECT_LE(prim.conflict, 1.0);
    }
}

TEST_F(PrimitivesTest, CovarianceValid) {
    auto points = createCluster(Eigen::Vector3d(1, 1, 1), 0.5, 100, 3);
    
    auto primitives = builder_->buildPrimitives(points);
    ASSERT_GT(primitives.size(), 0u);
    
    for (const auto& prim : primitives) {
        // Covariance should be symmetric positive definite
        auto cov = prim.covariance;
        
        // Symmetric check
        EXPECT_NEAR(cov(0, 1), cov(1, 0), 1e-6);
        EXPECT_NEAR(cov(0, 2), cov(2, 0), 1e-6);
        EXPECT_NEAR(cov(1, 2), cov(2, 1), 1e-6);
        
        // Positive eigenvalues
        auto eigenvalues = prim.getEigenvalues();
        EXPECT_GT(eigenvalues(0), 0);
        EXPECT_GT(eigenvalues(1), 0);
        EXPECT_GT(eigenvalues(2), 0);
    }
}

TEST_F(PrimitivesTest, UncertaintyWeighting) {
    std::vector<SemanticPoint> points;
    
    // Create points with varying uncertainty
    for (int i = 0; i < 50; ++i) {
        SemanticPoint p;
        p.position = Eigen::Vector3d(
            static_cast<double>(rand()) / RAND_MAX,
            static_cast<double>(rand()) / RAND_MAX,
            0);
        p.semantic_class = 1;
        p.semantic_confidence = 0.8f;
        p.uncertainty_semantic = 0.2;
        p.uncertainty_total = static_cast<double>(i) / 50.0;  // 0 to 1
        points.push_back(p);
    }
    
    auto primitives = builder_->buildPrimitives(points);
    EXPECT_GT(primitives.size(), 0u);
    
    // Low uncertainty points should have more influence on centroid
    // (This is a qualitative test)
}

TEST_F(PrimitivesTest, IncrementalUpdate) {
    config_.target_primitives = 1;
    builder_ = std::make_unique<GaussianPrimitiveBuilder>(config_);

    auto initial_points = createCluster(Eigen::Vector3d(0, 0, 0), 0.3, 50, 1);
    auto primitives = builder_->buildPrimitives(initial_points);
    
    ASSERT_GT(primitives.size(), 0u);
    auto initial_centroid = primitives[0].centroid;
    auto initial_id = primitives[0].id;
    
    // Add more points slightly offset
    auto new_points = createCluster(Eigen::Vector3d(0.5, 0, 0), 0.3, 50, 1);
    
    // Incremental update should preserve the primitive identity while moving it
    auto updated = builder_->updatePrimitives(primitives, new_points, 1.0);
    
    // Centroid should have moved
    ASSERT_GT(updated.size(), 0u);
    EXPECT_EQ(updated[0].id, initial_id);
    EXPECT_GT((updated[0].centroid - initial_centroid).norm(), 0.01);
}

TEST_F(PrimitivesTest, MergePrimitives) {
    // Create two similar primitives
    GaussianPrimitive p1, p2;
    p1.centroid = Eigen::Vector3d(0, 0, 0);
    p1.covariance = Eigen::Matrix3d::Identity() * 0.1;
    p1.semantic_class = 1;
    p1.point_count = 30;
    p1.total_weight = 30.0;
    p1.uncertainty = 0.2;
    p1.class_probabilities = {0.1, 0.8, 0.1};
    
    p2.centroid = Eigen::Vector3d(0.1, 0, 0);
    p2.covariance = Eigen::Matrix3d::Identity() * 0.1;
    p2.semantic_class = 1;
    p2.point_count = 20;
    p2.total_weight = 20.0;
    p2.uncertainty = 0.2;
    p2.class_probabilities = {0.1, 0.8, 0.1};

    auto merged = builder_->mergePrimitives(p1, p2);
    
    // Should be merged into one
    EXPECT_EQ(merged.point_count, 50);
    EXPECT_EQ(merged.semantic_class, 1);
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "test_primitives");
    return RUN_ALL_TESTS();
}
