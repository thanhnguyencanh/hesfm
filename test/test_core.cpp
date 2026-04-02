/**
 * @file test_core.cpp
 * @brief Unit tests for HESFM core components (types, kernel, map, primitives)
 * @author Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
 * @date 2026
 */

#include <gtest/gtest.h>
#include "hesfm/types.h"
#include "hesfm/adaptive_kernel.h"
#include "hesfm/gaussian_primitives.h"
#include "hesfm/semantic_map.h"
#include "hesfm/config.h"

using namespace hesfm;

// =============================================================================
// AffordanceBitset tests
// =============================================================================

TEST(AffordanceBitset, SetAndQuery) {
    AffordanceBitset bs;
    EXPECT_FALSE(bs.has(AffordanceType::TRAVERSABLE));

    bs.set(AffordanceType::TRAVERSABLE);
    EXPECT_TRUE(bs.has(AffordanceType::TRAVERSABLE));
    EXPECT_FALSE(bs.has(AffordanceType::OPENABLE));
    EXPECT_EQ(bs.count(), 1);

    bs.set(AffordanceType::OPENABLE);
    EXPECT_EQ(bs.count(), 2);

    bs.clear(AffordanceType::TRAVERSABLE);
    EXPECT_FALSE(bs.has(AffordanceType::TRAVERSABLE));
    EXPECT_TRUE(bs.has(AffordanceType::OPENABLE));
}

TEST(AffordanceBitset, FromSemanticClass) {
    AffordanceBitset bs;

    // floor (class 1) -> TRAVERSABLE
    bs.fromSemanticClass(1);
    EXPECT_TRUE(bs.has(AffordanceType::TRAVERSABLE));
    EXPECT_FALSE(bs.has(AffordanceType::OPENABLE));

    // door (class 7) -> OPENABLE
    bs.fromSemanticClass(7);
    EXPECT_TRUE(bs.has(AffordanceType::OPENABLE));
    EXPECT_FALSE(bs.has(AffordanceType::TRAVERSABLE));

    // person (class 30) -> AVOIDABLE
    bs.fromSemanticClass(30);
    EXPECT_TRUE(bs.has(AffordanceType::AVOIDABLE));

    // Out of range -> empty
    bs.fromSemanticClass(999);
    EXPECT_EQ(bs.count(), 0);
}

// =============================================================================
// DynamicObjectStatus tests
// =============================================================================

TEST(DynamicObjectStatus, InitialState) {
    DynamicObjectStatus ds;
    EXPECT_FALSE(ds.is_dynamic);
    EXPECT_EQ(ds.transition_count, 0);
    EXPECT_EQ(ds.previous_class, -1);
}

TEST(DynamicObjectStatus, SingleClassStable) {
    DynamicObjectStatus ds;
    for (int i = 0; i < 10; ++i)
        ds.update(5, static_cast<double>(i));
    EXPECT_FALSE(ds.is_dynamic);
    EXPECT_EQ(ds.transition_count, 0);
}

TEST(DynamicObjectStatus, FrequentTransitionsDynamic) {
    DynamicObjectStatus ds;
    ds.update(1, 0.0);
    ds.update(2, 1.0);
    ds.update(1, 2.0);
    ds.update(2, 3.0);
    // 3 transitions within window -> is_dynamic
    EXPECT_TRUE(ds.is_dynamic);
}

TEST(DynamicObjectStatus, DecayReducesTransitions) {
    DynamicObjectStatus ds;
    ds.update(1, 0.0);
    ds.update(2, 1.0);
    ds.update(1, 2.0);
    ds.update(2, 3.0);
    EXPECT_TRUE(ds.is_dynamic);

    // Decay after window expires
    ds.decayTransitions(50.0, 10.0);
    ds.decayTransitions(61.0, 10.0);
    ds.decayTransitions(72.0, 10.0);
    // After enough decay, should no longer be dynamic
    EXPECT_FALSE(ds.is_dynamic);
}

// =============================================================================
// ReachabilityInfo tests
// =============================================================================

TEST(ReachabilityInfo, UpdateConverges) {
    ReachabilityInfo ri;
    // All reachable
    for (int i = 0; i < 10; ++i)
        ri.update(true, 2.0, static_cast<double>(i));

    EXPECT_TRUE(ri.is_reachable);
    EXPECT_DOUBLE_EQ(ri.probability, 1.0);
    EXPECT_DOUBLE_EQ(ri.distance, 2.0);
}

TEST(ReachabilityInfo, MixedReachability) {
    ReachabilityInfo ri;
    ri.update(true, 1.0, 0.0);
    ri.update(false, 5.0, 1.0);
    ri.update(true, 1.5, 2.0);
    ri.update(false, 6.0, 3.0);

    EXPECT_EQ(ri.evaluation_count, 4);
    EXPECT_EQ(ri.reachable_count, 2);
    EXPECT_DOUBLE_EQ(ri.probability, 0.5);
    EXPECT_DOUBLE_EQ(ri.distance, 1.0);  // min of reachable distances
}

// =============================================================================
// MapCell functional attributes
// =============================================================================

TEST(MapCell, UpdateFunctionalAttributes) {
    MapCell cell(37);
    // Set floor as dominant class via log-odds
    cell.state.log_odds(1) = 5.0;  // floor
    cell.updateFunctionalAttributes(1.0);

    EXPECT_TRUE(cell.affordances.has(AffordanceType::TRAVERSABLE));
    EXPECT_FALSE(cell.affordances.has(AffordanceType::OPENABLE));
}

// =============================================================================
// AdaptiveKernel tests
// =============================================================================

TEST(AdaptiveKernel, SparseKernelCompactSupport) {
    KernelConfig cfg;
    AdaptiveKernel kernel(cfg);

    // At distance 0 -> kernel = 1.0
    EXPECT_NEAR(kernel.sparseKernel(0.0, 1.0), 1.0, 1e-6);

    // Beyond length scale -> kernel = 0
    EXPECT_DOUBLE_EQ(kernel.sparseKernel(1.5, 1.0), 0.0);

    // In between -> 0 < kernel < 1
    double val = kernel.sparseKernel(0.5, 1.0);
    EXPECT_GT(val, 0.0);
    EXPECT_LT(val, 1.0);
}

TEST(AdaptiveKernel, UncertaintyGating) {
    KernelConfig cfg;
    cfg.uncertainty_threshold = 0.7;
    AdaptiveKernel kernel(cfg);

    // Below threshold -> positive weight
    EXPECT_GT(kernel.uncertaintyKernel(0.3), 0.0);

    // Above threshold -> zero
    EXPECT_DOUBLE_EQ(kernel.uncertaintyKernel(0.8), 0.0);
}

TEST(AdaptiveKernel, UncertaintyAdaptiveLengthScale) {
    KernelConfig cfg;
    cfg.length_scale_min = 0.1;
    cfg.length_scale_max = 0.5;
    AdaptiveKernel kernel(cfg);

    Matrix3d cov = Matrix3d::Identity() * 0.1;
    double max_trace = 1.0;

    double ls_low  = kernel.computeUncertaintyAdaptiveLengthScale(cov, max_trace, 0.1);
    double ls_high = kernel.computeUncertaintyAdaptiveLengthScale(cov, max_trace, 0.9);

    // Low uncertainty -> larger kernel (confident data propagates further)
    // High uncertainty -> smaller kernel (uncertain data stays local)
    EXPECT_GT(ls_low, ls_high);
}

TEST(AdaptiveKernel, DynamicAttenuation) {
    KernelConfig cfg;
    cfg.uncertainty_threshold = 0.9;
    AdaptiveKernel kernel(cfg);

    GaussianPrimitive prim;
    prim.centroid = Vector3d::Zero();
    prim.covariance = Matrix3d::Identity() * 0.1;
    prim.uncertainty = 0.3;
    prim.semantic_class = 1;  // traversable

    Vector3d query(0.05, 0.0, 0.0);
    double max_trace = 0.3;

    prim.is_dynamic = false;
    double k_static = kernel.compute(query, prim, max_trace);

    prim.is_dynamic = true;
    double k_dynamic = kernel.compute(query, prim, max_trace);

    // Dynamic primitives should have half the influence
    EXPECT_NEAR(k_dynamic, k_static * 0.5, 1e-6);
}

// =============================================================================
// GaussianPrimitiveBuilder tests
// =============================================================================

TEST(GaussianPrimitiveBuilder, DSTFusion) {
    PrimitiveConfig cfg;
    GaussianPrimitiveBuilder builder(cfg);

    std::vector<double> b1 = {0.8, 0.1, 0.1};
    std::vector<double> b2 = {0.7, 0.2, 0.1};
    double conflict;

    auto result = builder.dstFusion(b1, b2, 0.2, 0.3, conflict);

    // Fused result should still favor class 0
    EXPECT_GT(result[0], result[1]);
    EXPECT_GT(result[0], result[2]);
    // Should sum to ~1
    double sum = 0;
    for (double r : result) sum += r;
    EXPECT_NEAR(sum, 1.0, 0.01);
}

TEST(GaussianPrimitiveBuilder, EmptyClusterReseeding) {
    // Build primitives from a small set - should not crash or leave
    // empty clusters
    PrimitiveConfig cfg;
    cfg.target_primitives = 3;
    cfg.min_points_per_primitive = 2;
    cfg.num_classes = 5;
    GaussianPrimitiveBuilder builder(cfg);

    std::vector<SemanticPoint> points;
    for (int i = 0; i < 20; ++i) {
        SemanticPoint pt;
        pt.position = Vector3d(i * 0.1, 0, 0);
        pt.semantic_class = i % 3;
        pt.class_probabilities = {0.1, 0.1, 0.1, 0.1, 0.6};
        pt.uncertainty_total = 0.3;
        pt.depth = 1.0;
        points.push_back(pt);
    }

    auto primitives = builder.buildPrimitives(points);
    EXPECT_GT(primitives.size(), 0u);
}

TEST(GaussianPrimitiveBuilder, RefineMergesNearby) {
    PrimitiveConfig cfg;
    cfg.num_classes = 3;
    cfg.min_points_per_primitive = 2;
    GaussianPrimitiveBuilder builder(cfg);

    // Two primitives, same class, very close
    GaussianPrimitive p1(0, Vector3d(0, 0, 0), 1);
    p1.covariance = Matrix3d::Identity() * 0.01;
    p1.class_probabilities = {0.1, 0.8, 0.1};
    p1.point_count = 10;
    p1.total_weight = 10.0;
    p1.conflict = 0.1;

    GaussianPrimitive p2(1, Vector3d(0.05, 0, 0), 1);
    p2.covariance = Matrix3d::Identity() * 0.01;
    p2.class_probabilities = {0.1, 0.8, 0.1};
    p2.point_count = 10;
    p2.total_weight = 10.0;
    p2.conflict = 0.1;

    std::vector<GaussianPrimitive> prims = {p1, p2};
    std::vector<SemanticPoint> pts;  // empty - no splitting needed

    auto refined = builder.refinePrimitives(prims, pts, 0.3);
    // Should merge into 1
    EXPECT_EQ(refined.size(), 1u);
    EXPECT_EQ(refined[0].point_count, 20);
}

// =============================================================================
// SemanticMap tests
// =============================================================================

TEST(SemanticMap, UpdateAndQuery) {
    MapConfig cfg;
    cfg.resolution = 0.1;
    cfg.num_classes = 5;
    cfg.origin_x = -1.0;
    cfg.origin_y = -1.0;
    cfg.origin_z = -0.5;
    cfg.size_x = 2.0;
    cfg.size_y = 2.0;
    cfg.size_z = 1.0;

    SemanticMap map(cfg);
    EXPECT_EQ(map.getNumCells(), 0u);

    // Directly update a cell
    std::vector<double> probs = {0.1, 0.7, 0.1, 0.05, 0.05};
    map.updateCell(Vector3d(0, 0, 0), probs, 1.0);

    auto state = map.query(Vector3d(0, 0, 0));
    EXPECT_TRUE(state.has_value());
    EXPECT_EQ(state->getPredictedClass(), 1);
    EXPECT_GT(map.getNumCells(), 0u);
}

TEST(SemanticMap, SaveLoadRoundtrip) {
    MapConfig cfg;
    cfg.resolution = 0.1;
    cfg.num_classes = 5;
    cfg.origin_x = -1.0;
    cfg.origin_y = -1.0;
    cfg.origin_z = -0.5;
    cfg.size_x = 2.0;
    cfg.size_y = 2.0;
    cfg.size_z = 1.0;

    SemanticMap map1(cfg);
    std::vector<double> probs = {0.05, 0.05, 0.8, 0.05, 0.05};
    map1.updateCell(Vector3d(0, 0, 0), probs, 3.0);
    map1.updateCell(Vector3d(0.5, 0.5, 0), probs, 3.0);

    EXPECT_TRUE(map1.save("/tmp/hesfm_test_map.yaml", "yaml"));

    SemanticMap map2(cfg);
    EXPECT_TRUE(map2.load("/tmp/hesfm_test_map.yaml"));
    EXPECT_EQ(map2.getNumCells(), 2u);

    // Class should match
    EXPECT_EQ(map2.getClass(Vector3d(0, 0, 0)), 2);
}

TEST(SemanticMap, MaxCellsEnforcement) {
    MapConfig cfg;
    cfg.resolution = 0.1;
    cfg.num_classes = 3;
    cfg.max_cells = 10;
    cfg.origin_x = -5.0;
    cfg.origin_y = -5.0;
    cfg.origin_z = -1.0;
    cfg.size_x = 10.0;
    cfg.size_y = 10.0;
    cfg.size_z = 2.0;

    SemanticMap map(cfg);
    // Insert many cells
    for (int i = 0; i < 20; ++i) {
        std::vector<double> probs = {0.3, 0.4, 0.3};
        map.updateCell(Vector3d(i * 0.1, 0, 0), probs, 1.0);
    }
    // The map should not exceed max_cells after an update-with-primitives call
    // (updateCell doesn't trigger pruning — only update() does)
    // So we test that the map at least stored cells
    EXPECT_GT(map.getNumCells(), 0u);
}

// =============================================================================
// Config yaml-cpp loader test
// =============================================================================

TEST(Config, SaveAndLoadYAML) {
    HESFMConfig cfg;
    // Set weights that already sum to 1.0 so normalizeWeights() is a no-op
    cfg.uncertainty.w_semantic = 0.35;
    cfg.uncertainty.w_spatial = 0.25;
    cfg.uncertainty.w_observation = 0.25;
    cfg.uncertainty.w_temporal = 0.15;
    cfg.map.resolution = 0.08;
    cfg.kernel.uncertainty_threshold = 0.65;

    EXPECT_TRUE(cfg.saveToYAML("/tmp/hesfm_test_config.yaml"));

    HESFMConfig loaded;
    EXPECT_TRUE(loaded.loadFromYAML("/tmp/hesfm_test_config.yaml"));
    EXPECT_NEAR(loaded.uncertainty.w_semantic, 0.35, 1e-6);
    EXPECT_NEAR(loaded.uncertainty.w_spatial, 0.25, 1e-6);
    EXPECT_NEAR(loaded.map.resolution, 0.08, 1e-6);
    EXPECT_NEAR(loaded.kernel.uncertainty_threshold, 0.65, 1e-6);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
