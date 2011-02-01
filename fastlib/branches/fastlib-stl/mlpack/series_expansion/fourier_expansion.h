/**
 * @file fourier_expansion.h
 *
 * This file contains a templatized class implementing the $O(p^D)$
 * Fourier expansion for computing the coefficients for a
 * far-field/local expansion for an arbitrary kernel function.
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef FOURIER_FARFIELD_EXPANSION
#define FOURIER_FARFIELD_EXPANSION

#include <fastlib/fastlib.h>

#include "kernel_aux.h"
#include "complex_matrix.h"
#include "fourier_series_expansion_aux.h"

/** @brief The far-field expansion class.
 */
template<typename TKernelAux, typename T = double>
class FourierExpansion {
  
 private:

  /** @brief The center of the expansion. */
  arma::Col<T> center_;
  
  /** @brief The coefficients. */
  ComplexVector<T> coeffs_;

  /** @brief The order of approximation. */
  int order_;
  
  /** auxilirary methods for the kernel (derivative, truncation error bound) */
  const TKernelAux *ka_;

  /** pointer to the kernel object inside kernel auxiliary object */
  const typename TKernelAux::TKernel *kernel_;

  /** pointer to the precomputed constants inside kernel auxiliary object */
  const FourierSeriesExpansionAux<double> *sea_;

 public:

  friend class FourierSeriesExpansionAux<T>;

  ////////// Public Type Representation //////////
  typedef T data_type;

  ////////// Getters/Setters //////////

  /** Get the coefficients */
  double bandwidth_sq() const { return kernel_->bandwidth_sq(); }
  
  /** Get the center of expansion */
  arma::Col<T>& get_center() { 
    return center_; 
  }

  const arma::Col<T>& get_center() const { 
    return center_; 
  }

  /** Get the coefficients */
  const ComplexVector<T>& get_coeffs() const {
    return coeffs_;
  }
  
  /** Get the approximation order */
  int get_order() const {
    return order_;
  }
  
  /** Get the maximum possible approximation order */
  int get_max_order() const {
    return sea_->get_max_order();
  }

  /** @brief Gets the weight sum.
   */
  T get_weight_sum() const {
    return coeffs_[0];
  }

  /** Set the approximation order */
  void set_order(int new_order) {
    order_ = new_order;
  }
  
  /** 
   * Set the center of the expansion - assumes that the center has been
   * initialized before...
   */
  void set_center(const arma::Col<T>& center) {
    center_ = center;
  }

  ////////// User-level Functions //////////
  
  /**
   * Accumulates the far field moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const arma::Mat<T>& data, const arma::Col<T>& weights,
			int begin, int end, int order) {
    
    // Loop over each data point and accumulate its contribution.
    index_t num_coefficients = sea_->get_total_num_coeffs(order);

    for(index_t i = begin; i < end; i++) {

      // The current data point.
      const double *point = data.colptr(i);

      for(index_t j = 0; j < num_coefficients; j++) {
	
	// Retrieve the current mapping.
	const std::vector<short int> &mapping = sea_->get_multiindex(j);
	
	// Dot product between the multiindex and the relative
	// coorindate of the data point.
	double dot_product = 0;
	for(index_t k = 0; k < mapping.size(); k++) {
	  dot_product += mapping[k] * (point[k] - center_[k]);
	}
	
	// For each coefficient, scale it and add to the current one.
	double trig_argument = 
	  -(sea_->integral_truncation_limit() * dot_product / 
	    (sea_->get_max_order() * sqrt(2 * bandwidth_sq())));
	std::complex<T> contribution(cos(trig_argument), sin(trig_argument));
	contribution *= weights[i];
	
	// This does the actual accumulation.
	coeffs_.set(j, coeffs_.get(j) + contribution);

      } // end of iterating over each coefficient position.
    }
  }

  /**
   * Refine the far field moment that has been computed before up to
   * a new order.
   */
  void RefineCoeffs(const arma::Mat<T>& data, const arma::Col<T>& weights,
		    int begin, int end, int order) {
    
    AccumulateCoeffs(data, weights, begin, end, order);
  }
  
  /**
   * Evaluates the far-field coefficients at the given point
   */
  T EvaluateField(const arma::Mat<T> &data, int row_num, int order) const {
    return EvaluateField(data.colptr(row_num), order);
  }
  
  T EvaluateField(const T *x_q, int order) const {
    return sea_->EvaluationOperator(*this, x_q, order);
  }
  
  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(const arma::Col<T>& center, const TKernelAux &ka) {

    // copy kernel type, center, and bandwidth squared
    kernel_ = &(ka.kernel_);
    center_ = center;
    order_ = -1;
    sea_ = &(ka.sea_);
    ka_ = &ka;
    
    // initialize coefficient array
    coeffs_.Init(sea_->get_max_total_num_coeffs());
    coeffs_.SetAll(0.0);
  }
  
  void Init(const TKernelAux &ka) {

    // copy kernel type, center, and bandwidth squared
    kernel_ = &(ka.kernel_);  
    order_ = -1;
    sea_ = &(ka.sea_);
    center_.Init(sea_->get_dimension());
    center_.SetZero();
    ka_ = &ka;
    
    // initialize coefficient array
    coeffs_.zeros(sea_->get_max_total_num_coeffs());
  }

  /** @brief Computes the required order for evaluating the far field
   *         expansion for any query point within the specified region
   *         for a given bound.
   */
  template<typename TBound>
  int OrderForEvaluating(const TBound &far_field_region,
			 const TBound &local_field_region,
			 double min_dist_sqd_regions,
			 double max_dist_sqd_regions,
			 double max_error, double *actual_error) const {
    
    // Fix me!
    return 3;
  }

  /** @brief Computes the required order for converting to the local
   *         expansion inside another region, so that the total error
   *         (truncation error of the far field expansion plus the
   *         conversion * error) is bounded above by the given user
   *         bound.
   *
   * @return the minimum approximation order required for the error,
   *         -1 if approximation up to the maximum order is not possible
   */
  template<typename TBound>
  int OrderForConvertingToLocal(const TBound &far_field_region,
				const TBound &local_field_region, 
				double min_dist_sqd_regions, 
				double max_dist_sqd_regions,
				double required_bound, 
				double *actual_error) const {

    // Fix me!
    return 3;
  }

  /** @brief Prints out the series expansion represented by this
   *         object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const {
  }

  /** @brief Translate from a far field expansion to the expansion
   *         here. The translated coefficients are added up to the
   *         ones here.
   */
  void TranslateFromFarField(const FourierExpansion &se) {
    
    sea_->TranslationOperator(se, *this, se.get_order());
  }
  
  /**
   * Translate to the given local expansion. The translated coefficients
   * are added up to the passed-in local expansion coefficients.
   */
  void TranslateToLocal(FourierExpansion<TKernelAux, T> &se, 
			int truncation_order) {
    
    sea_->TranslationOperator(*this, se, truncation_order);
  }

};

#endif
