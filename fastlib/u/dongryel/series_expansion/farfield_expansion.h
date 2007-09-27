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
  
  Vector &get_center() { return center_; }

  /** Get the coefficients */
  const Vector& get_coeffs() const { return coeffs_; }
  
  /** Get the approximation order */
  int get_order() const { return order_; }
  
  /** Get the maximum possible approximation order */
  int get_max_order() const { return sea_->get_max_order(); }

  /** Set the approximation order */
  void set_order(int new_order) { order_ = new_order; }
  
  /** 
   * Set the center of the expansion - assumes that the center has been
   * initialized before...
   */
  void set_center(const Vector &center) {
    
    for(index_t i = 0; i < center.length(); i++) {
      center_[i] = center[i];
    }
  }

  // interesting functions...
  
  /**
   * Accumulates the far field moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const Matrix& data, const Vector& weights,
			int begin, int end, int order);

  /**
   * Refine the far field moment that has been computed before up to
   * a new order.
   */
  void RefineCoeffs(const Matrix& data, const Vector& weights,
		    int begin, int end, int order);
  
  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateField(Matrix* data=NULL, int row_num=-1,
		       Vector* x_q=NULL) const;
  
  /**
   * Evaluates the three-way convolution with two other far field
   * expansions
   */
  double ConvolveField
    (const FarFieldExpansion<TKernel, TKernelDerivative> &fe2,
     const FarFieldExpansion<TKernel, TKernelDerivative> &fe3,
     int order1, int order2, int order3) const;

  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(double bandwidth, const Vector& center, 
	    SeriesExpansionAux *sea);

  void Init(double bandwidth, SeriesExpansionAux *sea);

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
(const Matrix& data, const Vector& weights, int begin, int end, 
 int order) {
  
  int dim = data.n_rows();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int r, i, j, k, t, tail;
  Vector heads;
  Vector x_r;
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
  Vector C_k;
    
  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
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
      double prod = weights[r] * tmp[i];
      
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
(const Matrix& data, const Vector& weights, int begin, int end, 
 int order) {
  
  if(order_ < 0) {
    
    AccumulateCoeffs(data, weights, begin, end, order);
    return;
  }

  int dim = data.n_rows();
  int old_total_num_coeffs = sea_->get_total_num_coeffs(order_);
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int r, i, j;
  Vector heads;
  Vector x_r;
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

  Vector C_k;

  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
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
      double prod = weights[r] * tmp[i];
      
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
double FarFieldExpansion<TKernel, TKernelDerivative>::ConvolveField
  (const FarFieldExpansion<TKernel, TKernelDerivative> &fe2,
   const FarFieldExpansion<TKernel, TKernelDerivative> &fe3,
   int order1, int order2, int order3) const {
  
  // bandwidth factor and multiindex mapping stuffs
  double bandwidth_factor = kd_.BandwidthFactor(bandwidth_sq());
  const ArrayList<int> *multiindex_mapping = sea_->get_multiindex_mapping();
  const ArrayList<int> *lower_mapping_index = sea_->get_lower_mapping_index();

  // get the total number of coefficients and coefficients
  int total_num_coeffs1 = sea_->get_total_num_coeffs(order1);
  int total_num_coeffs2 = sea_->get_total_num_coeffs(order2);
  int total_num_coeffs3 = sea_->get_total_num_coeffs(order3);
  int dim = sea_->get_dimension();
  Vector coeffs2, coeffs3;
  coeffs2.Alias(fe2.get_coeffs());
  coeffs3.Alias(fe3.get_coeffs());

  // actual accumulated sum
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;
  
  // some temporary
  double moment_i, moment_j, moment_k;

  // temporary array
  ArrayList<int> mu_nu_mapping;
  ArrayList<int> alpha_mu_eta_mapping;
  ArrayList<int> beta_gamma_nu_eta_mapping;
  ArrayList<int> alpha_mu_mapping;
  ArrayList<int> beta_nu_mapping;
  ArrayList<int> gamma_eta_mapping;
  alpha_mu_mapping.Init(dim);
  beta_nu_mapping.Init(dim);
  gamma_eta_mapping.Init(dim);
  mu_nu_mapping.Init(dim);
  alpha_mu_eta_mapping.Init(dim);
  beta_gamma_nu_eta_mapping.Init(dim);

  // partial derivatives table
  Matrix derivative_map_alpha;
  derivative_map_alpha.Init(dim, order1 + 1);
  Matrix derivative_map_beta;
  derivative_map_beta.Init(dim, order2 + 1);
  Matrix derivative_map_gamma;
  derivative_map_gamma.Init(dim, order3 + 1);
  
  // compute center differences and complete the table of partial derivatives
  Vector xI_xJ, xI_xK, xJ_xK;
  xI_xJ.Init(dim);
  xI_xK.Init(dim);
  xJ_xK.Init(dim);
  Vector xJ_center, xK_center;
  xJ_center.Alias(fe2.get_center());
  xK_center.Alias(fe3.get_center());

  for(index_t d = 0; d < dim; d++) {
    xI_xJ[d] = (center_[d] - xJ_center[d]) / bandwidth_factor;
    xI_xK[d] = (center_[d] - xK_center[d]) / bandwidth_factor;
    xJ_xK[d] = (xJ_center[d] - xK_center[d]) / bandwidth_factor;
  }
  kd_.ComputeDirectionalDerivatives(xI_xJ, derivative_map_alpha);
  kd_.ComputeDirectionalDerivatives(xI_xK, derivative_map_beta);
  kd_.ComputeDirectionalDerivatives(xJ_xK, derivative_map_gamma);

  // inverse factorials
  Vector inv_multiindex_factorials;
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // main loop
  for(index_t alpha = 0; alpha < total_num_coeffs1; alpha++) {

    ArrayList <int> alpha_mapping = multiindex_mapping[alpha];
    ArrayList <int> lower_mappings_for_alpha = lower_mapping_index[alpha];
    double alpha_derivative = kd_.ComputePartialDerivative
      (derivative_map_alpha, alpha_mapping);

    for(index_t mu = 0; mu < lower_mappings_for_alpha.size(); mu++) {

      ArrayList <int> mu_mapping = 
	multiindex_mapping[lower_mappings_for_alpha[mu]];

      // alpha - mu
      for(index_t d = 0; d < dim; d++) {
	alpha_mu_mapping[d] = alpha_mapping[d] - mu_mapping[d];
      }
      
      for(index_t beta = 0; beta < total_num_coeffs2; beta++) {
	
	ArrayList <int> beta_mapping = multiindex_mapping[beta];
	ArrayList <int> lower_mappings_for_beta = lower_mapping_index[beta];
	double beta_derivative = kd_.ComputePartialDerivative
	  (derivative_map_beta, beta_mapping);

	for(index_t nu = 0; nu < lower_mappings_for_beta.size(); nu++) {
	  
	  ArrayList<int> nu_mapping = 
	    multiindex_mapping[lower_mappings_for_beta[nu]];

	  // mu + nu and beta - nu
	  for(index_t d = 0; d < dim; d++) {
	    mu_nu_mapping[d] = mu_mapping[d] + nu_mapping[d];
	    beta_nu_mapping[d] = beta_mapping[d] - nu_mapping[d];
	  }

	  for(index_t gamma = 0; gamma < total_num_coeffs3; gamma++) {
	    
	    ArrayList <int> gamma_mapping = multiindex_mapping[gamma];
	    ArrayList <int> lower_mappings_for_gamma = 
	      lower_mapping_index[gamma];
	    double gamma_derivative = kd_.ComputePartialDerivative
	      (derivative_map_gamma, gamma_mapping);

	    for(index_t eta = 0; eta < lower_mappings_for_gamma.size(); eta++){
	      
	      // add up alpha, mu, eta and beta, gamma, nu, eta
	      int sign = 0;
	      
	      ArrayList<int> eta_mapping =
		multiindex_mapping[lower_mappings_for_gamma[eta]];

	      for(index_t d = 0; d < dim; d++) {
		alpha_mu_eta_mapping[d] = alpha_mapping[d] - mu_mapping[d] +
		  eta_mapping[d];
		beta_gamma_nu_eta_mapping[d] = beta_mapping[d] +
		  gamma_mapping[d] - nu_mapping[d] - eta_mapping[d];
		gamma_eta_mapping[d] = gamma_mapping[d] - eta_mapping[d];
		
		sign += 2 * (alpha_mapping[d] + beta_mapping[d] + 
			     gamma_mapping[d]) - mu_mapping[d] - nu_mapping[d] 
		  - eta_mapping[d];
	      }
	      if(sign % 2 == 1) {
		sign = -1;
	      }
	      else {
		sign = 1;
	      }
	      
	      // retrieve moments for appropriate multiindex maps
	      moment_i = 
		coeffs_[sea_->ComputeMultiindexPosition(mu_nu_mapping)];
	      moment_j = 
		coeffs2[sea_->ComputeMultiindexPosition(alpha_mu_eta_mapping)];
	      moment_k = 
		coeffs3[sea_->ComputeMultiindexPosition
			(beta_gamma_nu_eta_mapping)];

	      pos_sum += sign * 
		sea_->get_n_multichoose_k_by_pos
		(sea_->ComputeMultiindexPosition(mu_nu_mapping),
		 sea_->ComputeMultiindexPosition(mu_mapping)) *
		sea_->get_n_multichoose_k_by_pos
		(sea_->ComputeMultiindexPosition(alpha_mu_eta_mapping),
		 sea_->ComputeMultiindexPosition(eta_mapping)) *
		sea_->get_n_multichoose_k_by_pos
		(sea_->ComputeMultiindexPosition(beta_gamma_nu_eta_mapping),
		 sea_->ComputeMultiindexPosition(beta_nu_mapping)) *
		alpha_derivative * beta_derivative * gamma_derivative * 
		moment_i * moment_j * moment_k;

	    } // end of eta
	  } // end of gamma
	} // end of nu
      } // end of beta
    } // end of mu
  } // end of alpha
  
  // combine negative and positive sums
  sum = neg_sum + pos_sum;
  return sum;
}

template<typename TKernel, typename TKernelDerivative>
  void FarFieldExpansion<TKernel, TKernelDerivative>::Init
  (double bandwidth, const Vector& center, SeriesExpansionAux *sea) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_.Init(bandwidth);
  center_.Copy(center);
  order_ = -1;
  sea_ = sea;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernel, typename TKernelDerivative>
  void FarFieldExpansion<TKernel, TKernelDerivative>::Init
  (double bandwidth, SeriesExpansionAux *sea) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_.Init(bandwidth);
  center_.Init(sea->get_dimension());
  center_.SetZero();
  order_ = -1;
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

  // fix for multibody code
  int max_order = sea_->get_max_order() / 2;

  // find out the widest dimension and its length
  for(index_t d = 0; d < dim; d++) {
    DRange far_range = far_field_region.get(d);
    DRange local_range = local_field_region.get(d);
    max_ref_length += far_range.width() * far_range.width();
    max_query_length += local_range.width() * local_range.width();
  }
  max_ref_length = sqrt(max_ref_length) / 2.0;
  max_query_length = sqrt(max_query_length) / 2.0;

  double bandwidth = sqrt(kernel_.bandwidth_sq());
  double r_R = max_ref_length / bandwidth;
  double r_Q = max_query_length / bandwidth;

  int p_alpha = 0;
  double error = exp(-2 * r_R * r_Q);

  do {

    if(p_alpha > max_order) {
      return -1;
    }

    error *= r_Q * r_R * 2.0 / (p_alpha + 1.0);
    
    if(error <= max_error) {
      break;
    }
    p_alpha++;
  } while(1);

  *actual_error = error;
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
