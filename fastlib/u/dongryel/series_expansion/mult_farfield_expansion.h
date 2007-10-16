/**
 * @file mult_farfield_expansion.h
 *
 * The header file for the far field expansion in O(p^D) expansion
 */

#ifndef MULT_FARFIELD_EXPANSION
#define MULT_FARFIELD_EXPANSION

#include <values.h>

#include "fastlib/fastlib.h"
#include "kernel_aux.h"
#include "mult_series_expansion_aux.h"

template<typename TKernel, typename TKernelAux> 
class MultLocalExpansion;

/**
 * Far field expansion class
 */
template<typename TKernel, typename TKernelAux>
class MultFarFieldExpansion {
  FORBID_COPY(MultFarFieldExpansion);
  
 public:
  
  typedef TKernel Kernel;
  
  typedef TKernelAux KernelAux;

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
  MultSeriesExpansionAux *sea_;
  
  /** auxilirary methods for the kernel (derivative, truncation error bound) */
  KernelAux ka_;

 public:
  
  MultFarFieldExpansion() {}
  
  ~MultFarFieldExpansion() {}
  
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
		       Vector* x_q=NULL, int order=-1) const;
  
  /**
   * Evaluates the two-way convolution mixed with exhaustive computations
   * with two other far field expansions
   */
  double MixField(const Matrix &data, int node1_begin, int node1_end, 
		  int node2_begin, int node2_end,
		  const MultFarFieldExpansion<TKernel, TKernelAux> &fe2,
		  const MultFarFieldExpansion<TKernel, TKernelAux> &fe3,
		  int order2, int order3) const;

  /**
   * Evaluates the three-way convolution with two other far field
   * expansions
   */
  double ConvolveField
    (const MultFarFieldExpansion<TKernel, TKernelAux> &fe2,
     const MultFarFieldExpansion<TKernel, TKernelAux> &fe3,
     int order1, int order2, int order3) const;

  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(double bandwidth, const Vector& center, 
	    MultSeriesExpansionAux *sea);

  void Init(double bandwidth, MultSeriesExpansionAux *sea);

  /**
   * Computes the required order for evaluating the far field expansion
   * for any query point within the specified region for a given bound.
   */
  int OrderForEvaluating(const DHrectBound<2> &far_field_region,
			 const DHrectBound<2> &local_field_region,
			 double min_dist_sqd_regions,
			 double max_dist_sqd_regions,
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
  int OrderForConvertingToLocal(const DHrectBound<2> &far_field_region,
				const DHrectBound<2> &local_field_region, 
				double min_dist_sqd_regions, 
				double max_dist_sqd_regions,
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
  void TranslateFromFarField(const MultFarFieldExpansion &se);
  
  /**
   * Translate to the given local expansion. The translated coefficients
   * are added up to the passed-in local expansion coefficients.
   */
  void TranslateToLocal(MultLocalExpansion<TKernel, TKernelAux> &se);

};

template<typename TKernel, typename TKernelAux>
void MultFarFieldExpansion<TKernel, TKernelAux>::AccumulateCoeffs
(const Matrix& data, const Vector& weights, int begin, int end, 
 int order) {
  
  int dim = data.n_rows();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  int max_total_num_coeffs = sea_->get_max_total_num_coeffs();
  Vector x_r, tmp;
  double bandwidth_factor = ka_.BandwidthFactor(kernel_.bandwidth_sq());

  // initialize temporary variables
  x_r.Init(dim);
  tmp.Init(max_total_num_coeffs);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(max_total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(max_total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  
  // get the order of traversal for the given order of approximation
  ArrayList<int> &traversal_order = sea_->traversal_mapping_[order_];

  // Repeat for each reference point in this reference node.
  for(index_t r = begin; r < end; r++) {

    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(index_t i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
    }
    
    tmp.SetZero();
    tmp[0] = 1.0;

    for(index_t i = 1; i < total_num_coeffs; i++) {
      
      int index = traversal_order[i];
      ArrayList<int> &lower_mappings = sea_->lower_mapping_index_[index];

      // from the direct descendant, recursively compute the multipole moments
      int direct_ancestor_mapping_pos = 
	lower_mappings[lower_mappings.size() - 2];

      int position = 0;
      ArrayList<int> &mapping = sea_->multiindex_mapping_[index];
      ArrayList<int> &direct_ancestor_mapping = 
	sea_->multiindex_mapping_[direct_ancestor_mapping_pos];
      for(index_t i = 0; i < dim; i++) {
	if(mapping[i] != direct_ancestor_mapping[i]) {
	  position = i;
	  break;
	}
      }
      
      tmp[index] = tmp[direct_ancestor_mapping_pos] * x_r[position];
    }

    // Tally up the result in A_k.
    for(index_t i = 0; i < total_num_coeffs; i++) {

      int index = traversal_order[i];
      double prod = weights[r] * tmp[index];
      
      if(prod > 0) {
	pos_coeffs[index] += prod;
      }
      else {
	neg_coeffs[index] += prod;
      }
    }

  } // End of looping through each reference point

  for(index_t r = 0; r < total_num_coeffs; r++) {
    int index = traversal_order[r];
    coeffs_[index] += (pos_coeffs[index] + neg_coeffs[index]) * 
      sea_->inv_multiindex_factorials_[index];
  }
}

template<typename TKernel, typename TKernelAux>
void MultFarFieldExpansion<TKernel, TKernelAux>::RefineCoeffs
(const Matrix& data, const Vector& weights, int begin, int end, 
 int order) {

  // if we already have the order of approximation, then return.
  if(order_ >= order) {
    return;
  }

  // otherwise, recompute from scratch... this could be improved potentially
  // but I believe it will not squeeze out more performance (as in O(D^p)
  // expansions).
  else {
    order_ = order;
    
    coeffs_.SetZero();
    AccumulateCoeffs(data, weights, begin, end, order);
  }
}

template<typename TKernel, typename TKernelAux>
double MultFarFieldExpansion<TKernel, TKernelAux>::
  EvaluateField(Matrix* data, int row_num, Vector* x_q, int order) const {
  
  // dimension
  int dim = sea_->get_dimension();

  // total number of coefficients
  int total_num_coeffs = sea_->get_total_num_coeffs(order);

  // square root times bandwidth
  double bandwidth_factor = ka_.BandwidthFactor(kernel_.bandwidth_sq());
  
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
  ka_.ComputeDirectionalDerivatives(x_q_minus_x_R, derivative_map);
  
  // get the order of traversal for the given order of approximation
  ArrayList<int> &traversal_order = sea_->traversal_mapping_[order_];

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(index_t j = 0; j < total_num_coeffs; j++) {
    
    int index = traversal_order[j];
    ArrayList<int> mapping = sea_->get_multiindex(index);
    double arrtmp = ka_.ComputePartialDerivative(derivative_map, mapping);
    double prod = coeffs_[index] * arrtmp;
    
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

template<typename TKernel, typename TKernelAux>
  void MultFarFieldExpansion<TKernel, TKernelAux>::Init
  (double bandwidth, const Vector& center, MultSeriesExpansionAux *sea) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_.Init(bandwidth);
  center_.Copy(center);
  order_ = -1;
  sea_ = sea;

  // pass in the pointer to the kernel and the series expansion auxiliary
  // object
  ka_.kernel_ = &kernel_;
  ka_.msea_ = sea_;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernel, typename TKernelAux>
  void MultFarFieldExpansion<TKernel, TKernelAux>::Init
  (double bandwidth, MultSeriesExpansionAux *sea) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_.Init(bandwidth);
  center_.Init(sea->get_dimension());
  center_.SetZero();
  order_ = -1;
  sea_ = sea;

  // pass in the pointer to the kernel and the series expansion auxiliary
  // object
  ka_.kernel_ = &kernel_;
  ka_.msea_ = sea_;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernel, typename TKernelAux>
  int MultFarFieldExpansion<TKernel, TKernelAux>::OrderForEvaluating
  (const DHrectBound<2> &far_field_region, 
   const DHrectBound<2> &local_field_region, double min_dist_sqd_regions,
   double max_dist_sqd_regions, double max_error, double *actual_error) const {

  return ka_.OrderForEvaluatingMultFarField(far_field_region,
					    local_field_region,
					    min_dist_sqd_regions, 
					    max_dist_sqd_regions, max_error,
					    actual_error);
}

template<typename TKernel, typename TKernelAux>
  int MultFarFieldExpansion<TKernel, TKernelAux>::
  OrderForConvertingToLocal(const DHrectBound<2> &far_field_region,
			    const DHrectBound<2> &local_field_region, 
			    double min_dist_sqd_regions, 
			    double max_dist_sqd_regions,
			    double max_error, 
			    double *actual_error) const {

  return ka_.OrderForConvertingFromMultFarFieldToMultLocal
    (far_field_region, local_field_region, min_dist_sqd_regions,
     max_dist_sqd_regions, max_error, actual_error);
}

template<typename TKernel, typename TKernelAux>
  void MultFarFieldExpansion<TKernel, TKernelAux>::PrintDebug
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

template<typename TKernel, typename TKernelAux>
  void MultFarFieldExpansion<TKernel, TKernelAux>::TranslateFromFarField
  (const MultFarFieldExpansion &se) {
  
  double bandwidth_factor = ka_.BandwidthFactor(se.bandwidth_sq());
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
  if(order == 0) {
    return;
  }
  else {
    order_ = order;
  }
  
  // compute center difference
  for(index_t j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  // get the order of traversal for the given order of approximation
  ArrayList<int> &traversal_order = sea_->traversal_mapping_[order];

  for(index_t j = 0; j < total_num_coeffs; j++) {
   
    int index = traversal_order[j];
    ArrayList <int> gamma_mapping = multiindex_mapping[index];
    ArrayList <int> lower_mappings_for_gamma = lower_mapping_index[index];
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

template<typename TKernel, typename TKernelAux>
void MultFarFieldExpansion<TKernel, TKernelAux>::TranslateToLocal
  (MultLocalExpansion<TKernel, TKernelAux> &se) {
  
  Vector pos_arrtmp, neg_arrtmp;
  Matrix derivative_map;
  Vector local_center;
  Vector cent_diff;
  Vector local_coeffs;
  int local_order = se.get_order();
  int dimension = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);
  int limit;
  double bandwidth_factor = ka_.BandwidthFactor(se.bandwidth_sq());

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
  pos_arrtmp.Init(sea_->get_max_total_num_coeffs());
  neg_arrtmp.Init(sea_->get_max_total_num_coeffs());

  // compute center difference divided by bw_times_sqrt_two;
  for(index_t j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // compute required partial derivatives
  ka_.ComputeDirectionalDerivatives(cent_diff, derivative_map);
  ArrayList<int> beta_plus_alpha;
  beta_plus_alpha.Init(dimension);

  // get the order of traversal for the given order of approximation
  ArrayList<int> &traversal_order = sea_->traversal_mapping_[order_];

  for(index_t j = 0; j < total_num_coeffs; j++) {

    int index = traversal_order[j];
    ArrayList<int> beta_mapping = sea_->get_multiindex(index);
    pos_arrtmp[index] = neg_arrtmp[index] = 0;

    for(index_t k = 0; k < total_num_coeffs; k++) {

      int index_k = traversal_order[k];

      ArrayList<int> alpha_mapping = sea_->get_multiindex(index_k);
      for(index_t d = 0; d < dimension; d++) {
	beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
	ka_.ComputePartialDerivative(derivative_map, beta_plus_alpha);
      
      double prod = coeffs_[index_k] * derivative_factor;

      if(prod > 0) {
	pos_arrtmp[index] += prod;
      }
      else {
	neg_arrtmp[index] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  Vector C_k_neg = sea_->get_neg_inv_multiindex_factorials();
  for(index_t j = 0; j < total_num_coeffs; j++) {
    int index = traversal_order[j];
    local_coeffs[index] += (pos_arrtmp[index] + neg_arrtmp[index]) * 
      C_k_neg[index];
  }
}

#endif
