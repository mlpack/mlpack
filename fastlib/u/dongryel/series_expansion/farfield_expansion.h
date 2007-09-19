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
		       Vector* point=NULL);
  
  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(const TKernel& kernel, const Vector& center, 
	    SeriesExpansionAux *sea);

  /**
   * Computes the required order for evaluating the far field expansion
   * for any query point within the specified region for a given bound.
   */
  int OrderForEvaluating(const DHrectBound &far_field_region) const;

  /**
   * Computes the required order for converting to the local expansion
   * inside another region, so that the total error (truncation error
   * of the far field expansion plus the conversion error) is bounded
   * above by the given user bound.
   *
   * @return the minimum approximation order required for the error,
   *         -1 if approximation up to the maximum order is not possible
   */
  int OrderForConverting(const DHrectBound &far_field_region,
			 const DHrectBound &local_field_region, 
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
  void TranslateToLocal(LocalExpansion &se);

};

template<typename TKernel, typename TKernelDerivative>
void SeriesExpansion<TKernel, TKernelDerivative>::AccumulateCoeffs
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
void SeriesExpansion<TKernel, TKernelDerivative>::RefineCoeffs
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
double EvaluateField(Matrix* data=NULL, int row_num=-1,
		     Vector* point=NULL) {

    // dimension
  int dim = sea_->get_dimension();

  // total number of coefficients
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  // square root times bandwidth
  double bandwidth_factor = kd_.BandwidthFactor(kernel_.bandwidth_sq());
  
  // the evaluated sum
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

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2))
  for(index_t j = 0; j < total_num_coeffs; j++) {
    ArrayList<int> mapping = sea_->get_multiindex(j);
    arrtmp[j] = kd_.ComputePartialDerivative(derivative_map, mapping);
  }
  
  // tally up the multipole sum
  for(index_t j = 0; j < total_num_coeffs; j++) {
    multipole_sum += coeffs_[j] * arrtmp[j];
  }

  return multipole_sum;
}

#endif
