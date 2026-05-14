// =============================================================================
//  HESFM — Uncertainty-aware Bayesian Kernel Inference (UA-BKI)
//
//  Implements Eqs. (14)-(15) of:
//    Kim, Seo, Min, "Evidential Semantic Mapping in Off-road Environments
//    with Uncertainty-aware Bayesian Kernel Inference", IROS 2024.
//
//  Differences vs. standard S-BKI (Gan et al. 2020):
//    • Update uses the per-pixel Dirichlet *probability vector* p_i (not the
//      one-hot argmax label)  — Continuous Categorical likelihood.
//    • Kernel length scale is modulated by the per-point uncertainty u_i.
//    • Points with u_i above an adaptive threshold are skipped.
//
//  Author: HESFM @ JAIST
// =============================================================================
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace hesfm {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

// -----------------------------------------------------------------------------
//  Sparse kernel (Melkumyan & Ramos 2009; Eq. (7) in Kim et al.)
// -----------------------------------------------------------------------------
inline double sparse_kernel(double d, double l, double sigma0)
{
  if (d >= l) return 0.0;
  const double r = d / l;
  const double cs = std::cos(2.0 * M_PI * r);
  const double sn = std::sin(2.0 * M_PI * r);
  return sigma0 * ( (2.0 + cs) / 3.0 * (1.0 - r)
                  + (1.0 / (2.0 * M_PI)) * sn );
}

// -----------------------------------------------------------------------------
//  Uncertainty-aware adaptive kernel  (Eq. (15) in Kim et al.)
//
//      k(x*, x_i) = { k'(d, l*β*exp(1 - γ*u_i), σ0)   if u_i ≤ U_thr
//                   { 0                                otherwise
//
//  β  : base length-scale gain for confident predictions (recommend 0.5–1.0)
//  γ  : decay rate w.r.t. uncertainty                    (recommend 3.0–10.0)
//  U_thr : either fixed (e.g. 0.7) or adaptive (top-ũ% drop)
// -----------------------------------------------------------------------------
struct UABKIParams {
  double l_base    = 0.3;     // base length scale  [m]
  double sigma0    = 1.0;     // kernel scaling
  double beta      = 0.5;     // confidence gain
  double gamma     = 3.0;     // uncertainty decay
  double u_thr     = 0.7;     // fixed uncertainty cap
  double alpha0    = 1.0;     // Dirichlet prior pseudo-count per class
  bool   use_adaptive_thr = false;
  double drop_top_pct     = 0.10;  // when adaptive: drop the ũ% most uncertain
};

inline double ua_kernel(double d, double u, const UABKIParams& p)
{
  if (u > p.u_thr) return 0.0;
  const double l = p.l_base * p.beta * std::exp(1.0 - p.gamma * u);
  return sparse_kernel(d, std::max(l, 1e-3), p.sigma0);
}

// -----------------------------------------------------------------------------
//  Per-voxel Dirichlet posterior, with Continuous Categorical update.
//
//  α_t^c = α_{t-1}^c + Σ_i k(x*, x_i) * p_i^c
//
//  For C classes. We store α as a contiguous Eigen vector for speed.
// -----------------------------------------------------------------------------
class DirichletVoxel {
public:
  using VecXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;

  DirichletVoxel() = default;

  void init(int num_classes, double alpha0 = 1.0) {
    alpha_.setConstant(num_classes, alpha0);
  }

  // -- update with one observation (already-kernel-weighted) ---------------
  // weighted_p has length C and equals  k(x*, x_i) * p_i.
  void accumulate(const VecXd& weighted_p) {
    if (alpha_.size() == 0) alpha_.setZero(weighted_p.size());
    alpha_ += weighted_p;
  }

  // -- posterior mean ------------------------------------------------------
  VecXd mean() const {
    const double S = alpha_.sum();
    return alpha_ / std::max(S, 1e-9);
  }

  int    argmax_class() const {
    int idx; alpha_.maxCoeff(&idx); return idx;
  }

  double vacuity() const {
    const double S = alpha_.sum();
    return static_cast<double>(alpha_.size()) / std::max(S, 1e-9);
  }

  // Var[α^ψ_t] from Eq. (6) of S-BKI for the predicted class only.
  double predicted_class_variance() const {
    const double S = alpha_.sum();
    int psi; double a_psi = alpha_.maxCoeff(&psi);
    return a_psi * (S - a_psi) / (S * S * (S + 1.0));
  }

  // Confidence ∈ [0, 1] used in the paper's Brier score:
  //     Conf_i = 1 - Var[α^ψ_t] / Var_max
  double confidence(double var_max = 0.25) const {
    return std::clamp(1.0 - predicted_class_variance() / var_max, 0.0, 1.0);
  }

  const VecXd& alpha() const { return alpha_; }
  VecXd&       alpha()       { return alpha_; }

private:
  VecXd alpha_;
};

// -----------------------------------------------------------------------------
//  Single observation packet consumed by the BKI updater. Mirrors the fields
//  carried in the /hesfm/semantic_cloud topic.
// -----------------------------------------------------------------------------
struct EvidentialObservation {
  Eigen::Vector3d position;            // x_i in world frame
  Eigen::VectorXd probabilities;       // p_i, length C, sums to ~1
  double          uncertainty;         // u_i ∈ [0, 1]
};

// -----------------------------------------------------------------------------
//  Apply one BKI update over a list of observations whose voxels overlap
//  this voxel's neighbourhood (caller is responsible for pre-filtering by
//  a spatial query, e.g. an octree range search of radius l_base).
// -----------------------------------------------------------------------------
inline void update_voxel_ua_bki(
  DirichletVoxel&                              voxel,
  const Eigen::Vector3d&                       voxel_center,
  const std::vector<EvidentialObservation>&    obs,
  const UABKIParams&                           params,
  int                                          num_classes)
{
  if (voxel.alpha().size() != num_classes) voxel.init(num_classes, params.alpha0);

  // Optional: derive an adaptive threshold from this batch.
  // Adaptive mode drops the top `drop_top_pct` most uncertain observations
  // *in addition to* the fixed cap, so the effective threshold is the
  // stricter (smaller) of the two.
  double u_thr_eff = params.u_thr;
  if (params.use_adaptive_thr && !obs.empty()) {
    std::vector<double> us; us.reserve(obs.size());
    for (const auto& o : obs) us.push_back(o.uncertainty);
    std::sort(us.begin(), us.end());
    const double frac = std::clamp(1.0 - params.drop_top_pct, 0.0, 1.0);
    std::size_t k = static_cast<std::size_t>(std::ceil(frac * us.size()));
    if (k == 0) k = 1;
    if (k > us.size()) k = us.size();
    u_thr_eff = std::min(params.u_thr, us[k - 1]);
  }

  for (const auto& o : obs) {
    if (o.uncertainty > u_thr_eff) continue;
    const double d = (o.position - voxel_center).norm();

    UABKIParams p_eff = params;
    p_eff.u_thr = u_thr_eff;
    const double k = ua_kernel(d, o.uncertainty, p_eff);
    if (k <= 0.0) continue;

    // Eq. (14): α_t = α_{t-1} + k(·) * p_i
    voxel.accumulate(k * o.probabilities);
  }
}

}  // namespace hesfm
