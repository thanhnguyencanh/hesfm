# =============================================================================
# GaussianPrimitive.msg
# =============================================================================
# A Gaussian primitive aggregating multiple semantic points
# Core representation in the hierarchical HESFM framework
# G = (mu, Sigma, p, u) where:
#   mu = centroid, Sigma = covariance, p = class probs, u = uncertainty
# =============================================================================

# Unique identifier for this primitive
uint32 id

# -----------------------------------------------------------------------------
# Geometric Properties
# -----------------------------------------------------------------------------

# Centroid position (mean of aggregated points)
geometry_msgs/Point centroid

# Covariance matrix (3x3, stored as 9 elements in row-major order)
# [sigma_xx, sigma_xy, sigma_xz,
#  sigma_yx, sigma_yy, sigma_yz,
#  sigma_zx, sigma_zy, sigma_zz]
float64[9] covariance

# Eigenvalues of covariance matrix (for ellipsoid visualization)
# Sorted in descending order
float64[3] eigenvalues

# Eigenvectors as quaternion (orientation of ellipsoid)
geometry_msgs/Quaternion orientation

# -----------------------------------------------------------------------------
# Semantic Properties
# -----------------------------------------------------------------------------

# Most likely semantic class
uint8 semantic_class

# Aggregated class probability distribution (DST fusion result)
float32[] class_probabilities

# Aggregated uncertainty from constituent points
float32 uncertainty

# DST conflict measure (high conflict may trigger splitting)
# K = sum_{A ∩ B = ∅} m1(A) * m2(B)
float32 conflict

# -----------------------------------------------------------------------------
# Aggregation Statistics
# -----------------------------------------------------------------------------

# Number of points aggregated into this primitive
uint32 point_count

# Sum of point weights (inverse uncertainty)
float32 total_weight

# Adaptive kernel length scale for this primitive
# l = l_min + (l_max - l_min) * (trace(Sigma) / max_trace)^(1/3)
float64 length_scale

# -----------------------------------------------------------------------------
# Extended State
# -----------------------------------------------------------------------------

# Is this primitive associated with a dynamic object?
bool is_dynamic

# Reachability probability [0, 1]
float32 reachability

# Affordance values (application-specific)
float32[] affordances
