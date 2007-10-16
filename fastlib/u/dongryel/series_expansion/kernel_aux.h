/**
 * @file kernel_aux.h
 *
 * The header file for the class for computing auxiliary stuffs for the kernel
 * functions (derivative, truncation error bound)
 */

#ifndef KERNEL_AUX
#define KERNEL_AUX

#include "fastlib/fastlib.h"

#include "mult_series_expansion_aux.h"
#include "series_expansion_aux.h"

/**
 * Auxiliary computer class for Gaussian kernel
 */
class GaussianKernelAux {
  FORBID_COPY(GaussianKernelAux);

 public:

  /** pointer to the Gaussian kernel */
  GaussianKernel *kernel_;

  /** pointer to the series expansion auxiliary object */
  SeriesExpansionAux *sea_;

  MultSeriesExpansionAux *msea_;

  GaussianKernelAux() {}

  ~GaussianKernelAux() {}

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
				  ArrayList<int> mapping) const {
    
    double partial_derivative = 1.0;
    
    for(index_t d = 0; d < mapping.size(); d++) {
      partial_derivative *= derivative_map.get(d, mapping[d]);
    }
    return partial_derivative;
  }

  int OrderForEvaluatingFarField
    (const DHrectBound<2> &far_field_region, 
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    double frontfactor = 
      exp(-min_dist_sqd_regions / (4 * kernel_->bandwidth_sq()));
    double widest_width = 0;
    int dim = far_field_region.dim();
    int max_order = sea_->get_max_order();

    // find out the widest dimension and its length
    for(index_t d = 0; d < dim; d++) {
      DRange range = far_field_region.get(d);
      widest_width = max(widest_width, range.width());
    }
  
    double two_bandwidth = 2 * sqrt(kernel_->bandwidth_sq());
    double r = widest_width / two_bandwidth;

    // This is not really necessary for O(D^p) expansion, but it is for
    // speeding up the convergence of the Taylor expansion.
    if(r >= 1.0) {
      return -1;
    }

    double r_raised_to_p_alpha = 1.0;
    double ret;
    int p_alpha = 0;
    double floor_fact, ceil_fact;
    int remainder;

    do {

      if(p_alpha > max_order - 1)
	return -1;

      r_raised_to_p_alpha *= r;

      floor_fact = 
	sea_->factorial((int)floor(((double) p_alpha) / ((double) dim)));
      ceil_fact = 
	sea_->factorial((int)ceil(((double) p_alpha) / ((double) dim)));

      if(floor_fact < 0.0 || ceil_fact < 0.0) {
	return -1;
      }

      remainder = p_alpha % dim;

      ret = frontfactor * 
	(sea_->get_total_num_coeffs(p_alpha + 1) -
	 sea_->get_total_num_coeffs(p_alpha)) * r_raised_to_p_alpha /
	sqrt(pow(floor_fact, dim - remainder) * pow(ceil_fact, remainder));
    
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
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions, 
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    double max_ref_length = 0;
    double max_query_length = 0;
    int dim = sea_->get_dimension();

    for(index_t i = 0; i < dim; i++) {
      DRange far_field_range = far_field_region.get(i);
      DRange local_range = local_field_region.get(i);
      max_ref_length = max(max_ref_length, far_field_range.width());
      max_query_length = max(max_query_length, local_range.width());
    }
  
    double two_times_bandwidth = sqrt(kernel_->bandwidth_sq()) * 2;
    double r_R = max_ref_length / two_times_bandwidth;
    double r_Q = max_query_length / two_times_bandwidth;
    double sqrt_two_r_R = sqrt(2.0) * r_R;
    double sqrt_two_r_Q = sqrt(2.0) * r_Q;

    if(sqrt_two_r_R >= 1.0 || sqrt_two_r_Q >= 1.0) {
      return -1;
    }

    int p_alpha = -1;
    double sqrt_two_r_R_raised_to_p = 1.0;
    double r_Q_raised_to_p = 1.0;
    int remainder;
    double ret2;
    double frontfactor = 
      exp(-min_dist_sqd_regions / (4.0 * kernel_->bandwidth_sq()));
    double floor_fact, ceil_fact;

    do {
      p_alpha++;

      r_Q_raised_to_p *= r_Q;
      sqrt_two_r_R_raised_to_p *= sqrt_two_r_R;
      floor_fact = 
	sea_->factorial((int) floor((double) p_alpha / (double) dim));
      ceil_fact = 
	sea_->factorial((int) ceil((double)p_alpha / (double)dim));

      if(floor_fact < 0 || ceil_fact < 0 || 
	 p_alpha > sea_->get_max_order() - 1) {
	return -1;
      }

      remainder = p_alpha % dim;

      ret2 = (sea_->get_total_num_coeffs(p_alpha + 1) -
	      sea_->get_total_num_coeffs(p_alpha))
	/ sqrt(pow(floor_fact, dim - remainder) *
	       pow(ceil_fact, remainder));
      ret2 *= (r_Q_raised_to_p + sqrt_two_r_R_raised_to_p * 
	       sea_->get_total_num_coeffs(p_alpha)) * frontfactor;

    } while(ret2 >= max_error);

    *actual_error = ret2;
    return p_alpha;
  }
  
  int OrderForEvaluatingLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {
    
    double frontfactor =
      exp(-min_dist_sqd_regions / (4 * kernel_->bandwidth_sq()));
    double widest_width = 0;
    int dim = local_field_region.dim();
    int max_order = sea_->get_max_order();
  
    // find out the widest dimension and its length
    for(index_t d = 0; d < dim; d++) {
      DRange range = local_field_region.get(d);
      widest_width = max(widest_width, range.width());
    }
  
    double two_bandwidth = 2 * sqrt(kernel_->bandwidth_sq());
    double r = widest_width / two_bandwidth;
  
    // This is not really necessary for O(D^p) expansion, but it is for
    // speeding up the convergence of the Taylor expansion.
    if(r >= 1.0)
      return -1;
  
    double r_raised_to_p_alpha = 1.0;
    double ret;
    int p_alpha = 0;
    double floor_fact, ceil_fact;
    int remainder;
  
    do {
    
      if(p_alpha > max_order - 1)
	return -1;
    
      r_raised_to_p_alpha *= r;
    
      floor_fact =
	sea_->factorial((int)floor(((double) p_alpha) / ((double) dim)));
      ceil_fact =
	sea_->factorial((int)ceil(((double) p_alpha) / ((double) dim)));
    
      if(floor_fact < 0.0 || ceil_fact < 0.0)
	return -1;
    
      remainder = p_alpha % dim;
    
      ret = frontfactor *
	(sea_->get_total_num_coeffs(p_alpha + 1) -
	 sea_->get_total_num_coeffs(p_alpha)) * r_raised_to_p_alpha /
	sqrt(pow(floor_fact, dim - remainder) * pow(ceil_fact, remainder));
    
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

  int OrderForEvaluatingMultFarField
    (const DHrectBound<2> &far_field_region, 
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    return -1;
  }

  int OrderForConvertingFromMultFarFieldToMultLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions, 
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    return -1;
  }
  
  int OrderForEvaluatingMultLocal
    (const DHrectBound<2> &far_field_region,
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    return -1;
  }
};

/**
 * Auxilairy computer class for Epanechnikov kernel
 */
class EpanKernelAux {
  FORBID_COPY(EpanKernelAux);
  
 public:

  EpanKernel *kernel_;
  
  SeriesExpansionAux *sea_;

  EpanKernelAux() {}

  ~EpanKernelAux() {}

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
				  ArrayList<int> mapping) const {
    
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
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions,
     double max_dist_sqd_regions, double max_error, 
     double *actual_error) const {

    // first check that the maximum distances between the two regions are
    // within the bandwidth, otherwise the expansion is not valid
    if(max_dist_sqd_regions > kernel_->bandwidth_sq()) {
      return -1;
    }

    double widest_width = 0;
    int dim = far_field_region.dim();

    // find out the widest dimension and its length
    for(index_t d = 0; d < dim; d++) {
      DRange range = far_field_region.get(d);
      widest_width = max(widest_width, range.width());
    }
  
    // find out the max distance between query and reference region in L1
    // sense
    double farthest_distance_manhattan = 0;
    for(index_t d = 0; d < dim; d++) {
      DRange far_range = far_field_region.get(d);
      DRange local_range = local_field_region.get(d);
      double far_range_centroid_coord = far_range.lo + far_range.width() / 2;

      farthest_distance_manhattan =
	max(farthest_distance_manhattan,
	    max(fabs(far_range_centroid_coord - local_range.lo),
		fabs(far_range_centroid_coord - local_range.hi)));
    }

    // divide by the two times the bandwidth to find out how wide it is
    // in terms of the bandwidth
    double two_bandwidth = 2 * sqrt(kernel_->bandwidth_sq());
    double r = widest_width / two_bandwidth;
    farthest_distance_manhattan /= sqrt(kernel_->bandwidth_sq());

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
     const DHrectBound<2> &local_field_region, double min_dist_sqd_regions, 
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
};

#endif
