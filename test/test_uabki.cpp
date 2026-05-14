// Smoke test for uncertainty_aware_bki.h
// Compile with:
//   g++ -std=c++17 -I include test_uabki.cpp -o test_uabki
// Just verifies the header compiles and the math is sane.
#include "hesfm/uncertainty_aware_bki.h"
#include <iostream>
#include <cassert>

int main() {
  using namespace hesfm;

  // Test 1: sparse_kernel(0, l, sigma0) = sigma0
  double k0 = sparse_kernel(0.0, 1.0, 1.0);
  std::cout << "sparse_kernel(0, 1, 1) = " << k0 << "  (expect 1.0)\n";
  assert(std::abs(k0 - 1.0) < 1e-3);

  // Test 2: sparse_kernel(d >= l, l, sigma0) = 0
  double kfar = sparse_kernel(2.0, 1.0, 1.0);
  std::cout << "sparse_kernel(2, 1, 1) = " << kfar << "  (expect 0)\n";
  assert(kfar == 0.0);

  // Test 3: ua_kernel skips highly uncertain points
  UABKIParams p;
  p.u_thr = 0.5; p.l_base = 1.0;
  double k_unc_high = ua_kernel(0.1, 0.9, p);
  std::cout << "ua_kernel(d=0.1, u=0.9) = " << k_unc_high << "  (expect 0)\n";
  assert(k_unc_high == 0.0);

  // Test 4: confident prediction -> larger length scale -> higher kernel value
  double k_unc_low  = ua_kernel(0.1, 0.05, p);
  double k_unc_med  = ua_kernel(0.1, 0.30, p);
  std::cout << "ua_kernel(0.1, u=0.05) = " << k_unc_low
            << "  vs (0.1, u=0.30) = " << k_unc_med << "\n";
  assert(k_unc_low > k_unc_med);

  // Test 5: voxel update accumulates correctly
  DirichletVoxel v;
  v.init(4, 1.0);
  std::cout << "Initial vacuity: " << v.vacuity() << "  (expect 1.0)\n";
  assert(std::abs(v.vacuity() - 1.0) < 1e-3);

  Eigen::VectorXd p_obs(4);
  p_obs << 0.7, 0.1, 0.1, 0.1;
  v.accumulate(0.5 * p_obs);
  v.accumulate(0.5 * p_obs);

  std::cout << "After updates argmax = " << v.argmax_class() << "  (expect 0)\n";
  std::cout << "After updates vacuity = " << v.vacuity() << "\n";
  assert(v.argmax_class() == 0);
  assert(v.vacuity() < 1.0);

  // Test 6: full update path
  std::vector<EvidentialObservation> obs;
  for (int i = 0; i < 5; ++i) {
    EvidentialObservation o;
    o.position = Eigen::Vector3d(0.05 * i, 0, 0);
    o.probabilities = p_obs;
    o.uncertainty = 0.1 + 0.1 * i;               // 0.1, 0.2, 0.3, 0.4, 0.5
    obs.push_back(o);
  }
  DirichletVoxel v2;
  update_voxel_ua_bki(v2, Eigen::Vector3d::Zero(), obs, p, 4);
  std::cout << "Multi-obs voxel argmax: " << v2.argmax_class()
            << ", vacuity: " << v2.vacuity()
            << ", confidence: " << v2.confidence() << "\n";
  assert(v2.argmax_class() == 0);

  std::cout << "\nAll C++ tests passed.\n";
  return 0;
}
