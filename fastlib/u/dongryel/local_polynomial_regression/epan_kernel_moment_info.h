#ifndef EPAN_KERNEL_MOMENT_INFO_H
#define EPAN_KERNEL_MOMENT_INFO_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/bounds_aux.h"

class EpanKernelMomentInfo {
public:
  
  /** @brief $\sum_j \frac{w_j r_j}{h_j^2}$
   */
  Vector weighted_mass;
  
  /** @brief $\sum_j \frac{w_j ||r_j||^2}{h_j^2}$
   */
  double weighted_sumsq;
  
  /** @brief $\sum_j w_j$
   */
  double count;
  
  /** @brief $\sum_j \frac{w_j}{h_j^2}$
   */
  double weighted_count;
  
  /** @brief $\sum_j \frac{w_j}{h_j^4}$
   */
  double sum_weight_divided_by_bandwidth_quartic_pow;
  
  /** @brief $\sum_j \frac{w_j r_j}{h_j^4}$
   */
  Vector weighted_mass_divided_by_bandwidth_quartic_pow;
  
  /** @brief $\sum_j \frac{w_j ||r_j||^2}{h_j^4}$
   */
  double weighted_sumsq_divided_by_bandwidth_quartic_pow;
  
  /** @brief $\sum_j \frac{w_j r_j r_j^T}{h_j^4}$
   */
  Matrix weighted_outer_product;
  
  /** @brief $\sum_j \frac{w_j ||r_j||^2 r_j}{h_j^4}$
   */
  Vector magnitude_weighted_mass_divided_by_bandwidth_quartic_pow;
  
  /** @brief $\sum_j \frac{w_j ||r_j||^4}{h_j^4}$
   */
  double weighted_sumquartic;
  
public:
  
  EpanKernelMomentInfo() {}
  
  ~EpanKernelMomentInfo() {}
  
  void Init(index_t length) {
    weighted_mass.Init(length);
    weighted_mass_divided_by_bandwidth_quartic_pow.Init(length);
    weighted_outer_product.Init(length, length);
    magnitude_weighted_mass_divided_by_bandwidth_quartic_pow.
      Init(length);
    Reset();
  }
  
  void Reset() {
    weighted_mass.SetZero();
    weighted_sumsq = 0;
    count = 0;
    weighted_count = 0;
    sum_weight_divided_by_bandwidth_quartic_pow = 0;
    weighted_mass_divided_by_bandwidth_quartic_pow.SetZero();
    weighted_sumsq_divided_by_bandwidth_quartic_pow = 0;
    weighted_outer_product.SetZero();
    magnitude_weighted_mass_divided_by_bandwidth_quartic_pow.SetZero();
    weighted_sumquartic = 0;
  }
  
  void Add(double weight, double bandwidth_sq,
	   const Vector &reference_point) {
    
    double factor = weight / bandwidth_sq;
    double factor_div_bandwidth_quart = factor / bandwidth_sq;
    double reference_point_squared_length =
      la::Dot(reference_point, reference_point);

    la::AddExpert(factor, reference_point, &weighted_mass);
    weighted_sumsq += factor * reference_point_squared_length;
    count += weight;
    weighted_count += factor;

    sum_weight_divided_by_bandwidth_quartic_pow +=
      factor_div_bandwidth_quart;
    la::AddExpert(factor_div_bandwidth_quart, reference_point,
		  &weighted_mass_divided_by_bandwidth_quartic_pow);
    weighted_sumsq_divided_by_bandwidth_quartic_pow +=
      factor_div_bandwidth_quart * reference_point_squared_length;
	  
    for(index_t j = 0; j < weighted_outer_product.n_cols(); j++) {
      for(index_t i = 0; i < weighted_outer_product.n_rows(); i++) {
	weighted_outer_product.set
	  (i, j, weighted_outer_product.get(i, j) +
	   factor_div_bandwidth_quart * reference_point[i] *
	   reference_point[j]);
      }
    }
    la::AddExpert
      (factor_div_bandwidth_quart * reference_point_squared_length, 
       reference_point,
       &magnitude_weighted_mass_divided_by_bandwidth_quartic_pow);
    weighted_sumquartic += factor_div_bandwidth_quart * 
      reference_point_squared_length * reference_point_squared_length;
  }

  void Add(const EpanKernelMomentInfo& other) {
    la::AddTo(other.weighted_mass, &weighted_mass);
    weighted_sumsq += other.weighted_sumsq;
    count += other.count;
    weighted_count += other.weighted_count;
	  
    sum_weight_divided_by_bandwidth_quartic_pow +=
      other.sum_weight_divided_by_bandwidth_quartic_pow;
    la::AddTo(other.weighted_mass_divided_by_bandwidth_quartic_pow,
	      &weighted_mass_divided_by_bandwidth_quartic_pow);
    weighted_sumsq_divided_by_bandwidth_quartic_pow += 
      other.weighted_sumsq_divided_by_bandwidth_quartic_pow;
    la::AddTo(other.weighted_outer_product, &weighted_outer_product);
    la::AddTo
      (other.magnitude_weighted_mass_divided_by_bandwidth_quartic_pow,
       &magnitude_weighted_mass_divided_by_bandwidth_quartic_pow);
    weighted_sumquartic += other.weighted_sumquartic;
  }

  /**
   * Compute the squared kernel sum for a region of reference
   * points given the actual query point.
   */
  double ComputeSquaredKernelSum(const Vector& q) const {
    double kernel_sum = 2 * ComputeKernelSum(q) - count;
	  
    // The squared length of the query point.
    double q_sqd_length = la::Dot(q, q);
    double correction_term = 
      sum_weight_divided_by_bandwidth_quartic_pow * q_sqd_length * 
      q_sqd_length -
      4 * q_sqd_length * 
      la::Dot(q, weighted_mass_divided_by_bandwidth_quartic_pow) +
      2 * q_sqd_length *
      weighted_sumsq_divided_by_bandwidth_quartic_pow -
      4 * la::Dot
      (q, magnitude_weighted_mass_divided_by_bandwidth_quartic_pow) +
      weighted_sumquartic;
	  
    for(index_t j = 0; j < weighted_outer_product.n_cols(); j++) {
      for(index_t i = 0; i < weighted_outer_product.n_rows(); i++) {
	correction_term += 4 * q[j] * q[i] * 
	  weighted_outer_product.get(i, j);
      }
    }
    return kernel_sum + correction_term;
  }

  /**
   * Compute kernel sum for a region of reference points assuming we have
   * the actual query point.
   */
  double ComputeKernelSum(const Vector& q) const {
    double quadratic_term =
      weighted_count * la::Dot(q, q)
      - 2.0 * la::Dot(q, weighted_mass)
      + weighted_sumsq;
    return count - quadratic_term;
  }

  double ComputeKernelSum(const Vector &q, 
			  const Vector &reference_center,
			  double distance_squared,
			  double center_dot_center) const {

    double quadratic_term =
      (distance_squared - center_dot_center) * weighted_count
      + weighted_sumsq
      - 2 * la::Dot(q, weighted_mass)
      + 2 * weighted_count * la::Dot(q, reference_center);
	  
    return -quadratic_term + count;
  }
      
  template<int t_pow>
  double ComputeMinSquaredKernelSum
  (const DHrectBound<t_pow> query_bound) const {

    Vector center;
    double center_dot_center = 
      la::Dot(weighted_mass, weighted_mass) / weighted_count / 
      weighted_count;
	  
    DEBUG_ASSERT(weighted_count != 0);
	  
    center.Copy(weighted_mass);
    la::Scale(1.0 / weighted_count, &center);
	  
    Vector furthest_point_in_query_bound;
    furthest_point_in_query_bound.Init(weighted_mass.length());
    double furthest_dsqd;

    bounds_aux::MaxDistanceSq(query_bound, center,
			      furthest_point_in_query_bound,
			      furthest_dsqd);

    return ComputeSquaredKernelSum(furthest_point_in_query_bound,
				   center, furthest_dsqd, 
				   center_dot_center);
  }
      
  template<int t_pow>
  double ComputeMinKernelSum(const DHrectBound<t_pow> query_bound) 
    const {

    Vector center;
	  
    DEBUG_ASSERT(weighted_count != 0);
	  
    center.Copy(weighted_mass);
    la::Scale(1.0 / weighted_count, &center);
	  
    Vector furthest_point_in_query_bound;
    furthest_point_in_query_bound.Init(weighted_mass.length());
    double furthest_dsqd;

    bounds_aux::MaxDistanceSq(query_bound, center,
			      furthest_point_in_query_bound,
			      furthest_dsqd);

    return ComputeSquaredKernelSum(furthest_point_in_query_bound);
  }
};

#endif
