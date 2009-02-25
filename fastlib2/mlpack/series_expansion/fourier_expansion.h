/**
 * @file fourier_expansion.h
 *
 * This file contains a templatized class implementing the $O(p^D)$
 * Fourier expansion for computing the coefficients for a far-field
 * expansion for an arbitrary kernel function.
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef FOURIER_FARFIELD_EXPANSION
#define FOURIER_FARFIELD_EXPANSION


#include "fastlib/fastlib.h"
#include "kernel_aux.h"
#include "complex_matrix.h"
#include "fourier_series_expansion_aux.h"

/** @brief The far-field expansion class.
 */
template<typename TKernelAux, typename T = double>
class FourierExpansion {
  
 private:

  /** @brief The center of the expansion. */
  GenVector<T> center_;
  
  /** @brief The coefficients. */
  ComplexVector<T> coeffs_;

  /** @brief The order of approximation. */
  int order_;
  
  /** auxilirary methods for the kernel (derivative, truncation error bound) */
  const TKernelAux *ka_;

  /** pointer to the kernel object inside kernel auxiliary object */
  const typename TKernelAux::TKernel *kernel_;

  /** pointer to the precomputed constants inside kernel auxiliary object */
  const typename TKernelAux::TSeriesExpansionAux *sea_;

  OT_DEF(FourierExpansion) {
    OT_MY_OBJECT(center_);
    OT_MY_OBJECT(coeffs_);
    OT_MY_OBJECT(order_);
  }

 public:

  friend class FourierSeriesExpansionAux;

  ////////// Public Type Representation //////////
  typedef T data_type;

  ////////// Getters/Setters //////////

  /** Get the coefficients */
  double bandwidth_sq() const { return kernel_->bandwidth_sq(); }
  
  /** Get the center of expansion */
  Vector *get_center() { 
    return &center_; 
  }

  const Vector *get_center() const { 
    return &center_; 
  }

  /** Get the coefficients */
  const Vector& get_coeffs() const {
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
  void set_center(const GenVector<T> &center) {
    
    for(index_t i = 0; i < center.length(); i++) {
      center_[i] = center[i];
    }
  }

  ////////// User-level Functions //////////
  
  /**
   * Accumulates the far field moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const GenMatrix<T> &data, const GenVector<T> &weights,
			int begin, int end, int order) {
  }

  /**
   * Refine the far field moment that has been computed before up to
   * a new order.
   */
  void RefineCoeffs(const GenMatrix<T> &data, const GenVector<T> &weights,
		    int begin, int end, int order) {

  }
  
  /**
   * Evaluates the far-field coefficients at the given point
   */
  T EvaluateField(const GenMatrix<T> &data, int row_num, int order) const {
    return EvaluateField(data.GetColumnPtr(row_num), order);
  }
  
  T EvaluateField(const T *x_q, int order) const {
    return sea_->EvaluationOperator(*this, x_q, order);
  }
  
  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(const GenVector<T> &center, const TKernelAux &ka) {

    // copy kernel type, center, and bandwidth squared
    kernel_ = &(ka.kernel_);
    center_.Copy(center);
    order_ = -1;
    sea_ = &(ka.sea_);
    ka_ = &ka;
    
    // initialize coefficient array
    coeffs_.Init(sea_->get_max_total_num_coeffs());
    coeffs_.SetZero();
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
    coeffs_.Init(sea_->get_max_total_num_coeffs());
    coeffs_.SetZero();    
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
