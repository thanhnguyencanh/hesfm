/**
 * @file test_uncertainty.cpp
 * @brief Unit tests for HESFM uncertainty decomposition
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include "hesfm/uncertainty.h"
#include "hesfm/types.h"

using namespace hesfm;

class UncertaintyTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.w_semantic = 0.4;
        config_.w_spatial = 0.2;
        config_.w_observation = 0.25;
        config_.w_temporal = 0.15;
        config_.spatial_radius = 0.2;
        config_.min_neighbors = 3;
        
        decomposer_ = std::make_unique<UncertaintyDecomposer>(config_);
    }
    
    UncertaintyConfig config_;
    std::unique_ptr<UncertaintyDecomposer> decomposer_;
};

TEST_F(UncertaintyTest, SemanticUncertainty_Uniform) {
    // Uniform distribution should have maximum uncertainty
    std::vector<double> probs(40, 1.0 / 40.0);
    double u = decomposer_->computeSemanticUncertainty(probs);
    
    // EDL: K/S where K=40, S=sum(alpha)=40 for uniform
    EXPECT_NEAR(u, 1.0, 0.1);
}

TEST_F(UncertaintyTest, SemanticUncertainty_Confident) {
    // Confident prediction should have low uncertainty
    std::vector<double> probs(40, 0.001);
    probs[5] = 0.96;
    double u = decomposer_->computeSemanticUncertainty(probs);
    
    EXPECT_LT(u, 0.2);
}

TEST_F(UncertaintyTest, SpatialUncertainty_Consistent) {
    // All neighbors agree - low uncertainty
    std::vector<SemanticPoint> points;
    for (int i = 0; i < 10; ++i) {
        SemanticPoint p;
        p.position = Eigen::Vector3d(0.1 * i, 0, 0);
        p.semantic_class = 5;  // All same class
        points.push_back(p);
    }
    
    SemanticPoint query;
    query.position = Eigen::Vector3d(0.5, 0, 0);
    query.semantic_class = 5;
    
    double u = decomposer_->computeSpatialUncertainty(query, points);
    EXPECT_LT(u, 0.3);
}

TEST_F(UncertaintyTest, SpatialUncertainty_Inconsistent) {
    // Neighbors disagree - high uncertainty
    std::vector<SemanticPoint> points;
    for (int i = 0; i < 10; ++i) {
        SemanticPoint p;
        p.position = Eigen::Vector3d(0.1 * i, 0, 0);
        p.semantic_class = i % 5;  // Different classes
        points.push_back(p);
    }
    
    SemanticPoint query;
    query.position = Eigen::Vector3d(0.5, 0, 0);
    query.semantic_class = 0;
    
    double u = decomposer_->computeSpatialUncertainty(query, points);
    EXPECT_GT(u, 0.5);
}

TEST_F(UncertaintyTest, ObservationUncertainty_Close) {
    // Close observation with good density - low uncertainty
    SensorModel model;
    model.max_range = 6.0;
    model.sigma_range = 0.5;
    model.sigma_density = 0.3;
    model.sigma_angle = 0.2;
    
    double u = decomposer_->computeObservationUncertainty(1.0, 50.0, 0.0, model);
    EXPECT_LT(u, 0.3);
}

TEST_F(UncertaintyTest, ObservationUncertainty_Far) {
    // Far observation - higher uncertainty
    SensorModel model;
    model.max_range = 6.0;
    model.sigma_range = 0.5;
    model.sigma_density = 0.3;
    model.sigma_angle = 0.2;
    
    double u = decomposer_->computeObservationUncertainty(5.5, 10.0, 0.5, model);
    EXPECT_GT(u, 0.5);
}

TEST_F(UncertaintyTest, TotalUncertainty_Weighted) {
    // Test weighted combination
    UncertaintyDecomposition decomp;
    decomp.semantic = 0.3;
    decomp.spatial = 0.2;
    decomp.observation = 0.4;
    decomp.temporal = 0.1;
    
    double total = config_.w_semantic * decomp.semantic +
                   config_.w_spatial * decomp.spatial +
                   config_.w_observation * decomp.observation +
                   config_.w_temporal * decomp.temporal;
                   
    EXPECT_NEAR(total, 0.4 * 0.3 + 0.2 * 0.2 + 0.25 * 0.4 + 0.15 * 0.1, 1e-6);
}

TEST_F(UncertaintyTest, BatchProcessing) {
    std::vector<SemanticPoint> points;
    for (int i = 0; i < 100; ++i) {
        SemanticPoint p;
        p.position = Eigen::Vector3d(
            static_cast<double>(rand()) / RAND_MAX,
            static_cast<double>(rand()) / RAND_MAX,
            static_cast<double>(rand()) / RAND_MAX);
        p.semantic_class = rand() % 40;
        p.depth = 1.0 + 4.0 * static_cast<double>(rand()) / RAND_MAX;
        p.class_probabilities.resize(40, 0.01);
        p.class_probabilities[p.semantic_class] = 0.6;
        points.push_back(p);
    }
    
    Eigen::Vector3d sensor_origin(0, 0, 0);
    decomposer_->processBatch(points, sensor_origin);
    
    for (const auto& p : points) {
        EXPECT_GE(p.uncertainty_total, 0.0);
        EXPECT_LE(p.uncertainty_total, 1.0);
    }
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "test_uncertainty");
    return RUN_ALL_TESTS();
}
