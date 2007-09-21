/**
 * @file farfield_expansion.h
 *
 * The header file for the far field expansion
 */

#ifndef FARFIELD_EXPANSION
#define FARFIELD_EXPANSION

#include <values.h>

#include "fastlib/fastlib.h"
#include "kernel_derivative.h"
#include "series_expansion_aux.h"

template<typename TKernel, typename TKernelDerivative> 
class LocalExpansion;

/**
 * Far field expansion class
 */
template<typename TKernel, typename TKernelDerivative>
class FarFieldExpansion {
  FORBID_COPY(FarFieldExpansion);
  
 public:
  
  typedef TKernel Kernel;
  
  typedef TKernelDerivative KernelDerivative;

 private:
  
  /** The type of the kernel */
  Kernel kernel_;
  
  /** The center of the expansion */
  Vector center_;
  
  /** The coefficients */
  Vector coeffs_;
  
  /** order */
  int order_;
  
  /** precomputed quantities */
  SeriesExpansionAux *sea_;
  
  /** Derivative computer based on the kernel passed in */
  KernelDerivative kd_;

 public:
  
  FarFieldExpansion() {}
  
  ~FarFieldExpansion() {}
  
  // getters and setters
  
  /** Get the coefficients */
  double bandwidth_sq() const { return kernel_.bandwidth_sq(); }
  
  /** Get the center of expansion */
  const Vector& get_center() const { return center_; }
  
  /** Get the coefficients */
  const Vector& get_coeffs() const { return coeffs_; }
  
  /** Get the approximation order */
  int get_order() const { return order_; }
  
  /** Set the approximation order */
  void set_order(int new_order) { order_ = new_order; }
  
  // interesting functions...
  
  /**
   * Accumulates the far field moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const Matrix& data, const Vector& weights,
			const ArrayList<int>& rows, int order);

  /**
   * Refine the far field moment that has been computed before up to
   * a new order.
   */
  void RefineCoeffs(const Matrix& data, const Vector& weights,
		    const ArrayList<int>& rows, int order);
  
  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateField(Matrix* data=NULL, int row_num=-1,
		       Vector* x_q=NULL) const;
  
  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(double bandwidth, const Vector& center, 
	    SeriesExpansionAux *sea);

  /**
   * Computes the required order for evaluating the far field expansion
   * for any query point within the specified region for a given bound.
   */
  int OrderForEvaluating(const DHrectBound<2> &far_field_region,
			 double min_dist_sqd_regions,
			 double max_error, double *actual_error) const;

  /**
   * Computes the required order for converting to the local expansion
   * inside another region, so that the total error (truncation error
   * of the far field expansion plus the conversion error) is bounded
   * above by the given user bound.
   *
   * @return the minimum approximation order required for the error,
   *         -1 if approximation up to the maximum order is not possible
   */
  int OrderForConvertingtoLocal(const DHrectBound<2> &far_field_region,
				const DHrectBound<2> &local_field_region, 
				double min_dist_sqd_regions, 
				double required_bound, 
				double *actual_error) const;

  /**
   * Prints out the series expansion represented by this object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

  /**
   * Translate from a far field expansion to the expansion here.
   * The translated coefficients are added up to the ones here.
   */
  void TranslateFromFarField(const FarFieldExpansion &se);
  
  /**
   * Translate to the given local expansion. The translated coefficients
   * are added up to the passed-in local expansion coefficients.
   */
  void TranslateToLocal
    (LocalExpansion<TKernel, TKernelDerivative> &se);

};

template<typename TKernel, typename TKernelDerivative>
void FarFieldExpansion<TKernel, TKernelDerivative>::AccumulateCoeffs
(const Matrix& data, const Vector& weights, const ArrayList<int>& rows, 
 int order) {
  
  int dim = data.n_rows();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int num_rows = rows.size();
  int r, i, j, k, t, tail;
  Vector heads;
  Vector x_r;
  Vector C_k;
  double bandwidth_factor = kd_.BandwidthFactor(kernel_.bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  heads.Init(dim + 1);
  x_r.Init(dim);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
    
  // Repeat for each reference point in this reference node.
  for(r = 0; r < num_rows; r++) {
    
    // get the row number.
    int row_num = rows[r];
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, row_num) - center_[i]) / bandwidth_factor;
    }

    // initialize heads
    heads.SetZero();
    heads[dim] = MAXINT;
    
    tmp[0] = 1.0;
    
    for(k = 1, t = 1, tail = 1; k <= order; k++, tail = t) {
      for(i = 0; i < dim; i++) {
	int head = (int) heads[i];
	heads[i] = t;
	
	for(j = head; j < tail; j++, t++) {
	  tmp[t] = tmp[j] * x_r[i];
	}
      }
    }
    
    // Tally up the result in A_k.
    for(i = 0; i < total_num_coeffs; i++) {
      double prod = weights[row_num] * tmp[i];
      
      if(prod > 0) {
	pos_coeffs[i] += prod;
      }
      else {
	neg_coeffs[i] += prod;
      }
    }
    
  } // End of looping through each reference point

  // get multiindex factors
  C_k.Alias(sea_->get_inv_multiindex_factorials());

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernel, typename TKernelDerivative>
void FarFieldExpansion<TKernel, TKernelDerivative>::RefineCoeffs
(const Matrix& data, const Vector& weights, const ArrayList<int>& rows, 
 int order) {
  
  int dim = data.n_rows();
  int old_total_num_coeffs = sea_->get_total_num_coeffs(order_);
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int num_rows = rows.size();
  int r, i, j, k, t, tail;
  Vector heads;
  Vector x_r;
  Vector C_k;
  double bandwidth_factor = kd_.BandwidthFactor(kernel_.bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  heads.Init(dim + 1);
  x_r.Init(dim);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // if we already have the order of approximation, then return.
  if(order_ >= order) {
    return;
  }
  else {
    order_ = order;
  }
    
  // Repeat for each reference point in this reference node.
  for(r = 0; r < num_rows; r++) {
    
    // get the row number.
    int row_num = rows[r];
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, row_num) - center_[i]) / bandwidth_factor;
    }

    // compute in bruteforce way
    for(i = old_total_num_coeffs; i < total_num_coeffs; i++) {
      ArrayList<int> mapping = sea_->get_multiindex(i);
      tmp[i] = 1;

      for(j = 0; j < dim; j++) {
	tmp[i] *= pow(x_r[j], mapping[j]);
      }
    }
    
    // Tally up the result in A_k.
    for(i = 0; i < total_num_coeffs; i++) {
      double prod = weights[row_num] * tmp[i];
      
      if(prod > 0) {
	pos_coeffs[i] += prod;
      }
      else {
	neg_coeffs[i] += prod;
      }
    }
    
  } // End of looping through each reference point

  // get multiindex factors
  C_k.Alias(sea_->get_inv_multiindex_factorials());

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] = (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernel, typename TKernelDerivative>
double FarFieldExpansion<TKernel, TKernelDerivative>::
  EvaluateField(Matrix* data, int row_num, Vector* x_q) const {
  
  // dimension
  int dim = sea_->get_dimension();

  // total number of coefficients
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  // square root times bandwidth
  double bandwidth_factor = kd_.BandwidthFactor(kernel_.bandwidth_sq());
  
  // the evaluated sum
  double pos_multipole_sum = 0;
  double neg_multipole_sum = 0;
  double multipole_sum = 0;
  
  // computed derivative map
  Matrix derivative_map;
  derivative_map.Init(dim, order_ + 1);

  // temporary variable
  Vector arrtmp;
  arrtmp.Init(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  Vector x_q_minus_x_R;
  x_q_minus_x_R.Init(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(index_t d = 0; d < dim; d++) {
    if(x_q == NULL) {
      x_q_minus_x_R[d] = (data->get(d, row_num) - center_[d]) / 
	bandwidth_factor;
    }
    else {
      x_q_minus_x_R[d] = ((*x_q)[d] - center_[d]) / bandwidth_factor;
    }
  }

  // compute deriative maps based on coordinate difference.
  kd_.ComputeDirectionalDerivatives(x_q_minus_x_R, derivative_map);

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(index_t j = 0; j < total_num_coeffs; j++) {
    ArrayList<int> mapping = sea_->get_multiindex(j);
    double arrtmp = kd_.ComputePartialDerivative(derivative_map, mapping);
    double prod = coeffs_[j] * arrtmp;
    
    if(prod > 0) {
      pos_multipole_sum += prod;
    }
    else {
      neg_multipole_sum += prod;
    }
  }

  multipole_sum = pos_multipole_sum + neg_multipole_sum;
  return multipole_sum;
}

template<typename TKernel, typename TKernelDerivative>
  void FarFieldExpansion<TKernel, TKernelDerivative>::Init
  (double bandwidth, const Vector& center, SeriesExpansionAux *sea) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_.Init(bandwidth);
  center_.Copy(center);
  order_ = 0;
  sea_ = sea;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernel, typename TKernelDerivative>
  int FarFieldExpansion<TKernel, TKernelDerivative>::OrderForEvaluating
  (const DHrectBound<2> &far_field_region, double min_dist_sqd_regions,
   double max_error, double *actual_error) const {

  double frontfactor = 
    exp(-min_dist_sqd_regions / (4 * kernel_.bandwidth_sq()));
  double widest_width = 0;
  int dim = far_field_region.dim();
  int max_order = sea_->get_max_order();

  // find out the widest dimension and its length
  for(index_t d = 0; d < dim; d++) {
    DRange range = far_field_region.get(d);
    widest_width = max(widest_width, range.width());
  }
  
  double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
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

    if(p_alpha > max_order)
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

template<typename TKernel, typename TKernelDerivative>
  int FarFieldExpansion<TKernel, TKernelDerivative>::
  OrderForConvertingtoLocal(const DHrectBound<2> &far_field_region,
			    const DHrectBound<2> &local_field_region, 
			    double min_dist_sqd_regions, 
			    double max_error, 
			    double *actual_error) const {

  double max_ref_length = 0;
  double max_query_length = 0;

  int dim = far_field_region.dim();
  int max_order = sea_->get_max_order();

  // find out the widest dimension and its length
  for(index_t d = 0; d < dim; d++) {
    DRange far_range = far_field_region.get(d);
    DRange local_range = local_field_region.get(d);
    max_ref_length = max(max_ref_length, far_range.width());
    max_query_length = max(max_query_length, local_range.width());
  }

  double bwsqd = kernel_.bandwidth_sq();
  double two_times_bandwidth = sqrt(bwsqd) * 2;
  double r_R = max_ref_length / two_times_bandwidth;
  double r_Q = max_query_length / two_times_bandwidth;
  double sqrt_two_r_R = sqrt(2.0) * r_R;
  double sqrt_two_r_Q = sqrt(2.0) * r_Q;

  if(sqrt_two_r_R >= 1.0 || sqrt_two_r_Q >= 1.0)
    return -1;

  int p_alpha = 0;
  double sqrt_two_r_R_raised_to_p = 1.0;
  double r_Q_raised_to_p = 1.0;
  int remainder;
  double ret;
  double frontfactor = exp(-min_dist_sqd_regions / (4.0 * bwsqd));
  double floor_fact, ceil_fact;

  do {
    r_Q_raised_to_p *= r_Q;
    sqrt_two_r_R_raised_to_p *= sqrt_two_r_R;
    floor_fact = 
      sea_->factorial((int) floor((double) p_alpha / (double) dim));
    ceil_fact = 
      sea_->factorial((int) ceil((double)p_alpha / (double)dim));

    if(floor_fact < 0 || ceil_fact < 0 || p_alpha > max_order)
      return -1;

    remainder = p_alpha % dim;

    ret = (sea_->get_total_num_coeffs(p_alpha + 1) -
	   sea_->get_total_num_coeffs(p_alpha))
      / sqrt(pow(floor_fact, dim - remainder) *
             pow(ceil_fact, remainder));
    ret *= (r_Q_raised_to_p + sqrt_two_r_R_raised_to_p *
	    sea_->get_total_num_coeffs(p_alpha)) * frontfactor;
    
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

template<typename TKernel, typename TKernelDerivative>
  void FarFieldExpansion<TKernel, TKernelDerivative>::PrintDebug
  (const char *name, FILE *stream) const {

    
  int dim = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Far field expansion\n");
  fprintf(stream, "Center: ");
  
  for (index_t i = 0; i < center_.length(); i++) {
    fprintf(stream, "%g ", center_[i]);
  }
  fprintf(stream, "\n");
  
  fprintf(stream, "f(");
  for(index_t d = 0; d < dim; d++) {
    fprintf(stream, "x_q%d", d);
    if(d < dim - 1)
      fprintf(stream, ",");
  }
  fprintf(stream, ") = \\sum\\limits_{x_r \\in R} K(||x_q - x_r||) = ");
  
  for (index_t i = 0; i < total_num_coeffs; i++) {
    ArrayList<int> mapping = sea_->get_multiindex(i);
    fprintf(stream, "%g ", coeffs_[i]);
    
    fprintf(stream, "(-1)^(");
    for(index_t d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);
      if(d < dim - 1)
	fprintf(stream, " + ");
    }
    fprintf(stream, ") D^((");
    for(index_t d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);
      
      if(d < dim - 1)
	fprintf(stream, ",");
    }
    fprintf(stream, ")) f(x_q - x_R)");
    if(i < total_num_coeffs - 1) {
      fprintf(stream, " + ");
    }
  }
  fprintf(stream, "\n");
}

template<typename TKernel, typename TKernelDerivative>
  void FarFieldExpansion<TKernel, TKernelDerivative>::TranslateFromFarField
  (const FarFieldExpansion &se) {
  
  double bandwidth_factor = kd_.BandwidthFactor(se.bandwidth_sq());
  int dim = sea_->get_dimension();
  int order = se.get_order();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector prev_coeffs;
  Vector prev_center;
  const ArrayList < int > *multiindex_mapping = sea_->get_multiindex_mapping();
  const ArrayList < int > *lower_mapping_index = 
    sea_->get_lower_mapping_index();

  ArrayList <int> tmp_storage;
  Vector center_diff;
  Vector inv_multiindex_factorials;

  center_diff.Init(dim);

  // retrieve coefficients to be translated and helper mappings
  prev_coeffs.Alias(se.get_coeffs());
  prev_center.Alias(se.get_center());
  tmp_storage.Init(sea_->get_dimension());
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // no coefficients can be translated
  if(order == 0)
    return;
  else
    order_ = order;
  
  // compute center difference
  for(index_t j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  for(index_t j = 0; j < total_num_coeffs; j++) {
    
    ArrayList <int> gamma_mapping = multiindex_mapping[j];
    ArrayList <int> lower_mappings_for_gamma = lower_mapping_index[j];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(index_t k = 0; k < lower_mappings_for_gamma.size(); k++) {

      ArrayList <int> inner_mapping = 
	multiindex_mapping[lower_mappings_for_gamma[k]];

      int flag = 0;
      double diff1;
      
      // compute gamma minus alpha
      for(index_t l = 0; l < dim; l++) {
	tmp_storage[l] = gamma_mapping[l] - inner_mapping[l];

	if(tmp_storage[l] < 0) {
	  flag = 1;
	  break;
	}
      }
      
      if(flag) {
	continue;
      }
      
      diff1 = 1.0;
      
      for(index_t l = 0; l < dim; l++) {
	diff1 *= pow(center_diff[l] / bandwidth_factor, tmp_storage[l]);
      }

      double prod = prev_coeffs[lower_mappings_for_gamma[k]] * diff1 * 
	inv_multiindex_factorials
	[sea_->ComputeMultiindexPosition(tmp_storage)];
      
      if(prod > 0) {
	pos_coeff += prod;
      }
      else {
	neg_coeff += prod;
      }

    } // end of k-loop
    
    coeffs_[j] += pos_coeff + neg_coeff;

  } // end of j-loop
}

template<typename TKernel, typename TKernelDerivative>
void FarFieldExpansion<TKernel, TKernelDerivative>::TranslateToLocal
  (LocalExpansion<TKernel, TKernelDerivative> &se) {
  
  Vector pos_arrtmp, neg_arrtmp;
  Matrix derivative_map;
  Vector local_center;
  Vector cent_diff;
  Vector local_coeffs;
  int local_order = se.get_order();
  int dimension = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);
  int limit;
  double bandwidth_factor = kd_.BandwidthFactor(se.bandwidth_sq());

  // get center and coefficients for local expansion
  local_center.Alias(se.get_center());
  local_coeffs.Alias(se.get_coeffs());
  cent_diff.Init(dimension);

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(local_order < order_) {
    se.set_order(order_);
  }

  // compute Gaussian derivative
  limit = 2 * order_ + 1;
  derivative_map.Init(dimension, limit);
  pos_arrtmp.Init(total_num_coeffs);
  neg_arrtmp.Init(total_num_coeffs);

  // compute center difference divided by bw_times_sqrt_two;
  for(index_t j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // compute required partial derivatives
  kd_.ComputeDirectionalDerivatives(cent_diff, derivative_map);
  ArrayList<int> beta_plus_alpha;
  beta_plus_alpha.Init(dimension);

  for(index_t j = 0; j < total_num_coeffs; j++) {

    ArrayList<int> beta_mapping = sea_->get_multiindex(j);
    pos_arrtmp[j] = neg_arrtmp[j] = 0;

    for(index_t k = 0; k < total_num_coeffs; k++) {

      ArrayList<int> alpha_mapping = sea_->get_multiindex(k);
      for(index_t d = 0; d < dimension; d++) {
	beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
	kd_.ComputePartialDerivative(derivative_map, beta_plus_alpha);
      
      double prod = coeffs_[k] * derivative_factor;

      if(prod > 0) {
	pos_arrtmp[j] += prod;
      }
      else {
	neg_arrtmp[j] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  Vector C_k_neg = sea_->get_neg_inv_multiindex_factorials();
  for(index_t j = 0; j < total_num_coeffs; j++) {
    local_coeffs[j] += (pos_arrtmp[j] + neg_arrtmp[j]) * C_k_neg[j];
  }
}

#endif
