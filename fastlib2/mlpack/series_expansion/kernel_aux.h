/**
 * @file kernel_aux.h
 *
 * The header file for the class for computing auxiliary stuffs for the kernel
 * functions (derivative, truncation error bound)
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef KERNEL_AUX
#define KERNEL_AUX

#include "fastlib/fastlib.h"

#include "farfield_expansion.h"
#include "local_expansion.h"
#include "mult_farfield_expansion.h"
#include "mult_local_expansion.h"

/**
 * Auxiliary class for multiplicative p^D expansion for Gaussian
 * kernel.
 */
class GaussianKernelMultAux {

 public:

  typedef GaussianKernel TKernel;

  typedef MultSeriesExpansionAux TSeriesExpansionAux;
  
  typedef MultFarFieldExpansion<GaussianKernelMultAux> TFarFieldExpansion;
  
  typedef MultLocalExpansion<GaussianKernelMultAux> TLocalExpansion;

  /** pointer to the Gaussian kernel */
  TKernel kernel_;

  /** pointer to the series expansion auxiliary object */
  TSeriesExpansionAux sea_;

  OT_DEF_BASIC(GaussianKernelMultAux) {
    OT_MY_OBJECT(kernel_);
    OT_MY_OBJECT(sea_);
  }

 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
    sea_.Init(max_order, dim);
  }

  double BandwidthFactor(double bandwidth_sq) const {
    return sqrt(2 * bandwidth_sq);
  }

  void ComputeDirectionalDerivatives(const Vector &x, 
				     Matrix &derivative_map) const {
    
    int dim = derivative_map.n_rows();
    int order = derivative_map.n_cols() - 1;
    
    // precompute necessary Hermite polynomials based on coordinate difference
    for(index_t d = 0; d < dim; d++) {
      
      double coord_div_band = x[d];
      double d2 = 2 * coord_div_band;
      double facj = exp(-coord_div_band * coord_div_band);
      
      derivative_map.set(d, 0, facj);
      
      if(order > 0) {
	
	derivative_map.set(d, 1, d2 * facj);
	
	if(order > 1) {
	  for(index_t k = 1; k < order; k++) {
	    int k2 = k * 2;
	    derivative_map.set(d, k + 1, d2 * derivative_map.get(d, k) -
			       k2 * derivative_map.get(d, k - 1));
	  }
	}
      }
    } // end of looping over each dimension
  }

  double ComputePartialDerivative(const Matrix &derivative_map,
				  const ArrayList<int> &mapping) const {
    
    double partial_derivative = 1.0;
    
    for(index_t d = 0; d < mapping.size(); d++) {
      partial_derivative *= derivative_map.get(d, mapping[d]);
    }
    return partial_derivative;
  }

  int OrderForEvaluatingFarField
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {
    
    double max_far_field_length = 0;

    for(index_t d = 0; d < sea_.get_dimension(); d++) {
      const DRange &far_range = far_field_region.get(d);
      max_far_field_length = max(max_far_field_length, far_range.width());
    }

    double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
    double r = max_far_field_length / two_times_bandwidth;

    int dim = sea_.get_dimension();
    double r_raised_to_p_alpha = 1.0;
    double ret, ret2;
    int p_alpha = 0;
    double factorialvalue = 1.0;
    double first_factor, second_factor;
    double one_minus_r;

    // In this case, it is "impossible" to prune for the Gaussian kernel.
    if(r >= 1.0) {
      return -1;
    }
    one_minus_r = 1.0 - r;
    ret = 1.0 / pow(one_minus_r, dim);
  
    do {
      factorialvalue *= (p_alpha + 1);

      if(factorialvalue < 0.0 || p_alpha > sea_.get_max_order()) {
	return -1;
      }

      r_raised_to_p_alpha *= r;
      first_factor = 1.0 - r_raised_to_p_alpha;
      second_factor = r_raised_to_p_alpha / sqrt(factorialvalue);

      ret2 = ret * (pow((first_factor + second_factor), dim) -
		    pow(first_factor, dim));

      if(ret2 <= max_error) {
	break;
      }
      
      p_alpha++;

    } while(1);

    *actual_error = ret2;
    return p_alpha;
  }

  int OrderForConvertingFromFarFieldToLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions, 
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    double max_far_field_length = 0;
    double max_local_field_length = 0;

    for(index_t d = 0; d < sea_.get_dimension(); d++) {
      const DRange &far_range = far_field_region.get(d);
      const DRange &local_range = local_field_region.get(d);
      max_far_field_length = max(max_far_field_length, far_range.width());
      max_local_field_length = max(max_local_field_length, 
				   local_range.width());
    }

    double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
    double r = max_far_field_length / two_times_bandwidth;
    double r2 = max_local_field_length / two_times_bandwidth;

    int dim = sea_.get_dimension();
    double r_raised_to_p_alpha = 1.0;
    double ret, ret2;
    int p_alpha = 0;
    double factorialvalue = 1.0;
    double first_factor, second_factor;
    double one_minus_two_r, two_r;

    // In this case, it is "impossible" to prune for the Gaussian kernel.
    if(r >= 0.5 || r2 >= 0.5) {
      return -1;
    }

    r = max(r, r2);
    two_r = 2.0 * r;
    one_minus_two_r = 1.0 - two_r;
    ret = 1.0 / pow(one_minus_two_r * one_minus_two_r, dim);
  
    do {
      factorialvalue *= (p_alpha + 1);

      if(factorialvalue < 0.0 || p_alpha > sea_.get_max_order()) {
	return -1;
      }

      r_raised_to_p_alpha *= two_r;
      first_factor = 1.0 - r_raised_to_p_alpha;
      first_factor *= first_factor;
      second_factor = r_raised_to_p_alpha * (2.0 - r_raised_to_p_alpha)
	/ sqrt(factorialvalue);

      ret2 = ret * (pow((first_factor + second_factor), dim) -
		    pow(first_factor, dim));

      if(ret2 <= max_error) {
	break;
      }
      
      p_alpha++;

    } while(1);

    *actual_error = ret2;
    return p_alpha;
  }
  
  int OrderForEvaluatingLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {
        
    double max_local_field_length = 0;

    for(index_t d = 0; d < sea_.get_dimension(); d++) {
      const DRange &local_range = local_field_region.get(d);
      max_local_field_length = max(max_local_field_length, 
				   local_range.width());
    }

    double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
    double r = max_local_field_length / two_times_bandwidth;

    int dim = sea_.get_dimension();
    double r_raised_to_p_alpha = 1.0;
    double ret, ret2;
    int p_alpha = 0;
    double factorialvalue = 1.0;
    double first_factor, second_factor;
    double one_minus_r;

    // In this case, it is "impossible" to prune for the Gaussian kernel.
    if(r >= 1.0) {
      return -1;
    }
    one_minus_r = 1.0 - r;
    ret = 1.0 / pow(one_minus_r, dim);
  
    do {
      factorialvalue *= (p_alpha + 1);

      if(factorialvalue < 0.0 || p_alpha > sea_.get_max_order()) {
	return -1;
      }

      r_raised_to_p_alpha *= r;
      first_factor = 1.0 - r_raised_to_p_alpha;
      second_factor = r_raised_to_p_alpha / sqrt(factorialvalue);

      ret2 = ret * (pow((first_factor + second_factor), dim) -
		    pow(first_factor, dim));

      if(ret2 <= max_error) {
	break;
      }
      
      p_alpha++;

    } while(1);

    *actual_error = ret2;
    return p_alpha;
  }
  
  double UpperBoundOnDerivative(double min_dist_sqd_regions, int order) const {
    
    return exp(-min_dist_sqd_regions / (4 * kernel_.bandwidth_sq())) *
      pow(2, order / 2.0) * sea_.factorial(order);
  }
};

/** @brief Auxiliary class for Gaussian kernel
 */
class GaussianKernelAux {

 public:

  typedef GaussianKernel TKernel;

  typedef SeriesExpansionAux TSeriesExpansionAux;

  typedef FarFieldExpansion<GaussianKernelAux> TFarFieldExpansion;
  
  typedef LocalExpansion<GaussianKernelAux> TLocalExpansion;

  /** pointer to the Gaussian kernel */
  TKernel kernel_;

  /** pointer to the series expansion auxiliary object */
  TSeriesExpansionAux sea_;

  OT_DEF_BASIC(GaussianKernelAux) {
    OT_MY_OBJECT(kernel_);
    OT_MY_OBJECT(sea_);
  }

 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
    sea_.Init(max_order, dim);
  }

  double BandwidthFactor(double bandwidth_sq) const {
    return sqrt(2 * bandwidth_sq);
  }

  void ComputeDirectionalDerivatives(const Vector &x, 
				     Matrix &derivative_map) const {
    
    int dim = derivative_map.n_rows();
    int order = derivative_map.n_cols() - 1;
    
    // precompute necessary Hermite polynomials based on coordinate difference
    for(index_t d = 0; d < dim; d++) {
      
      double coord_div_band = x[d];
      double d2 = 2 * coord_div_band;
      double facj = exp(-coord_div_band * coord_div_band);
      
      derivative_map.set(d, 0, facj);
      
      if(order > 0) {
	
	derivative_map.set(d, 1, d2 * facj);
	
	if(order > 1) {
	  for(index_t k = 1; k < order; k++) {
	    int k2 = k * 2;
	    derivative_map.set(d, k + 1, d2 * derivative_map.get(d, k) -
			       k2 * derivative_map.get(d, k - 1));
	  }
	}
      }
    } // end of looping over each dimension
  }

  double ComputePartialDerivative(const Matrix &derivative_map,
				  const ArrayList<int> &mapping) const {
    
    double partial_derivative = 1.0;
    
    for(index_t d = 0; d < mapping.size(); d++) {
      partial_derivative *= derivative_map.get(d, mapping[d]);
    }
    return partial_derivative;
  }

  int OrderForEvaluatingFarField
    (const DHrectBound<2> &far_field_region, 
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    double frontfactor = 
      exp(-min_dist_sqd_regions / (4 * kernel_.bandwidth_sq()));
    double widest_width = 0;
    int dim = far_field_region.dim();
    int max_order = sea_.get_max_order();

    // find out the widest dimension and its length
    for(index_t d = 0; d < dim; d++) {
      const DRange &range = far_field_region.get(d);
      widest_width += range.width();
    }
  
    double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
    double r = widest_width / two_bandwidth;

    double r_raised_to_p_alpha = 1.0;
    double ret;
    int p_alpha = 0;

    do {

      if(p_alpha > max_order - 1) {
	return -1;
      }

      r_raised_to_p_alpha *= r;
      frontfactor /= sqrt(p_alpha + 1);
      
      ret = frontfactor * r_raised_to_p_alpha;
    
      if(ret > max_error) {
	p_alpha++;
      }
      else {
	break;
      }
    } while(1);

    *actual_error = ret;
    return p_alpha;
  }

  int OrderForConvertingFromFarFieldToLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions, 
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    double max_ref_length = 0;
    double max_query_length = 0;
    int dim = sea_.get_dimension();

    for(index_t i = 0; i < dim; i++) {
      const DRange &far_field_range = far_field_region.get(i);
      const DRange &local_range = local_field_region.get(i);
      max_ref_length += far_field_range.width();
      max_query_length += local_range.width();
    }
  
    double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
    double r_R = max_ref_length / two_times_bandwidth;
    double r_Q = max_query_length / two_times_bandwidth;

    int p_alpha = -1;
    double r_Q_raised_to_p = 1.0;
    double r_R_raised_to_p = 1.0;
    double ret2;
    double frontfactor =
      exp(-min_dist_sqd_regions / (4.0 * kernel_.bandwidth_sq()));
    double first_factorial = 1.0;
    double second_factorial = 1.0;
    double r_Q_raised_to_p_cumulative = 1;

    do {
      p_alpha++;

      if(p_alpha > sea_.get_max_order() - 1) {
	return -1;
      }

      first_factorial *= (p_alpha + 1);
      if(p_alpha > 0) {
	second_factorial *= sqrt(2 * p_alpha * (2 * p_alpha + 1));
      }
      r_Q_raised_to_p *= r_Q;
      r_R_raised_to_p *= r_R;
      
      ret2 = frontfactor * 
	(1.0 / first_factorial * r_R_raised_to_p * second_factorial * 
	 r_Q_raised_to_p_cumulative +
	 1.0 / sqrt(first_factorial) * r_Q_raised_to_p);

      r_Q_raised_to_p_cumulative += r_Q_raised_to_p / 
	((p_alpha > 0) ? (first_factorial / (p_alpha + 1)):first_factorial);

    } while(ret2 >= max_error);

    *actual_error = ret2;
    return p_alpha;
  }
  
  int OrderForEvaluatingLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {
    
    double frontfactor =
      exp(-min_dist_sqd_regions / (4 * kernel_.bandwidth_sq()));
    double widest_width = 0;
    int dim = local_field_region.dim();
    int max_order = sea_.get_max_order();
  
    // find out the widest dimension and its length
    for(index_t d = 0; d < dim; d++) {
      const DRange &range = local_field_region.get(d);
      widest_width += range.width();
    }
  
    double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
    double r = widest_width / two_bandwidth;

    double r_raised_to_p_alpha = 1.0;
    double ret;
    int p_alpha = 0;
  
    do {
    
      if(p_alpha > max_order - 1) {
	return -1;
      }
    
      r_raised_to_p_alpha *= r;
      frontfactor /= sqrt(p_alpha + 1);
    
      ret = frontfactor * r_raised_to_p_alpha;
    
      if(ret > max_error) {
	p_alpha++;
      }
      else {
	break;
      }
    } while(1);
  
    *actual_error = ret;
    return p_alpha;
  }

  double UpperBoundOnDerivative(double min_dist_sqd_regions, int order) const {

    // Please implement me...
    return DBL_MAX;
  }
};

/**
 * Auxilairy computer class for Epanechnikov kernel
 */
class EpanKernelAux {
  
 public:

  typedef EpanKernel TKernel;

  typedef SeriesExpansionAux TSeriesExpansionAux;

  typedef FarFieldExpansion<EpanKernelAux> TFarFieldExpansion;
  
  typedef LocalExpansion<EpanKernelAux> TLocalExpansion;

  TKernel kernel_;
  
  TSeriesExpansionAux sea_;

  OT_DEF_BASIC(EpanKernelAux) {
    OT_MY_OBJECT(kernel_);
    OT_MY_OBJECT(sea_);
  }

 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
    sea_.Init(max_order, dim);
  }

  double BandwidthFactor(double bandwidth_sq) const {
    return sqrt(bandwidth_sq);
  }

  void ComputeDirectionalDerivatives(const Vector &x, 
				     Matrix &derivative_map) const {

    int dim = derivative_map.n_rows();
    int order = derivative_map.n_cols() - 1;
    
    // precompute necessary Hermite polynomials based on coordinate difference
    for(index_t d = 0; d < dim; d++) {
      
      double coord_div_band = x[d];
      
      derivative_map.set(d, 0, coord_div_band * coord_div_band);

      if(order > 0) {
	derivative_map.set(d, 1, 2 * coord_div_band);
	
	if(order > 1) {
	  derivative_map.set(d, 2, -2);

	  for(index_t k = 3; k <= order; k++) {
	    derivative_map.set(d, k, 0);
	  }
	}
      }

    } // end of looping over each dimension
  }

  double ComputePartialDerivative(const Matrix &derivative_map,
				  const ArrayList<int> &mapping) const {
    
    int nonzero_count = 0;
    int nonzero_index = 0;

    // this is for checking whether the mapping represents a 
    // mixed partial derivative, which zero for Epanechnikov Kernel
    for(index_t d = 0; d < mapping.size(); d++) {
      if(mapping[d] > 0) {
	nonzero_count++;
	nonzero_index = d;
      }

      if(nonzero_count > 1) {
	return 0;
      }
    }
    
    // if it is not a mixed partial derivative, then compute
    if(nonzero_count == 0) {
      double prod = 0;
      for(index_t d = 0; d < mapping.size(); d++) {
	prod += derivative_map.get(d, 0);
      }
      return 1.0 - prod;
    }
    
    return derivative_map.get(nonzero_index, mapping[nonzero_index]);
  }

  int OrderForEvaluatingFarField
    (const DHrectBound<2> &far_field_region, 
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    // first check that the maximum distances between the two regions are
    // within the bandwidth, otherwise the expansion is not valid
    if(max_dist_sqd_regions > kernel_.bandwidth_sq()) {
      return -1;
    }

    double widest_width = 0;
    int dim = far_field_region.dim();

    // find out the widest dimension and its length
    for(index_t d = 0; d < dim; d++) {
      const DRange &range = far_field_region.get(d);
      widest_width = max(widest_width, range.width());
    }
  
    // find out the max distance between query and reference region in L1
    // sense
    double farthest_distance_manhattan = 0;
    for(index_t d = 0; d < dim; d++) {
      const DRange &far_range = far_field_region.get(d);
      const DRange &local_range = local_field_region.get(d);
      double far_range_centroid_coord = far_range.lo + far_range.width() / 2;

      farthest_distance_manhattan =
	max(farthest_distance_manhattan,
	    max(fabs(far_range_centroid_coord - local_range.lo),
		fabs(far_range_centroid_coord - local_range.hi)));
    }

    // divide by the two times the bandwidth to find out how wide it is
    // in terms of the bandwidth
    double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
    double r = widest_width / two_bandwidth;
    farthest_distance_manhattan /= sqrt(kernel_.bandwidth_sq());

    // try the 0-th order approximation first
    double error = 2 * dim * farthest_distance_manhattan * r;
    if(error < max_error) {
      *actual_error = error;
      return 0;
    }

    // try the 1st order approximation later
    error = dim * r * r;
    if(error < max_error) {
      *actual_error = error;
      return 1;
    }

    // failing all above, take up to 2nd terms
    *actual_error = 0;
    return 2;
  }

  int OrderForConvertingFromFarFieldToLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions, 
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    // currently, disabled but might be worth putting in...
    return -1;
  }
  
  int OrderForEvaluatingLocal
    (const DHrectBound<2> &far_field_region, 
     const DHrectBound<2> &local_field_region, 
     double min_dist_sqd_regions, double max_dist_sqd_regions, 
     double max_error, double *actual_error) const {

    // currrently disabled buy might be worth putting in
    return -1;
  }

  double UpperBoundOnDerivative(double min_dist_sqd_regions, int order) const {
    
    // Please implement me...
    return DBL_MAX;
  }
};

#endif
