/**
 * @file inverse_pow_dist_farfield_expansion.h
 *
 * This file contains a templatized class implementing a
 * three-dimensional spherical harmonic expansion for computing the
 * coefficients for a far-field expansion for the inverse power
 * distance kernel function.
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef INVERSE_POW_DIST_FARFIELD_EXPANSION
#define INVERSE_POW_DIST_FARFIELD_EXPANSION

#include <complex>
#include <values.h>

#include "fastlib/fastlib.h"
#include "kernel_aux.h"
#include "inverse_pow_dist_series_expansion_aux.h"

/** @brief The far field expansion class for the inverse power
 *         distance function.
 */
class InversePowDistFarFieldExpansion {
  
 private:

  /** @brief The inverse power factor.
   */
  double inverse_power_;

  /** @brief The center of the expansion. */
  Vector center_;
  
  /** @brief The coefficients. */
  ArrayList<GenMatrix<std::complex<double> > > coeffs_;
  
  /** @brief The order of approximation. */
  int order_;

  /** @brief The pointer to the precomputed constants inside kernel
   *         auxiliary object.
   */
  const InversePowDistSeriesExpansionAux *sea_;

  OT_DEF(InversePowDistFarFieldExpansion) {
    OT_MY_OBJECT(inverse_power_);
    OT_MY_OBJECT(center_);
    OT_MY_OBJECT(coeffs_);
    OT_MY_OBJECT(order_);
  }

 public:
  
  // getters and setters
  
  /** Get the center of expansion */
  Vector *get_center() { return &center_; }

  const Vector *get_center() const { return &center_; }
  
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

  void Accumulate(const double *v, double weight, int order);
  
  /**
   * Accumulates the far field moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const Matrix& data, const Vector& weights,
			int begin, int end, int order);
  
  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateField(const Matrix& data, int row_num, int order) const;
  double EvaluateField(const double *x_q, int order) const;
  
  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(const Vector& center, InversePowDistSeriesExpansionAux *sea);

  /** @brief Computes the required order for evaluating the far field
   *         expansion for any query point within the specified region
   *         for a given bound.
   */
  template<typename TBound>
  int OrderForEvaluating(const TBound &far_field_region,
			 const TBound &local_field_region,
			 double min_dist_sqd_regions,
			 double max_dist_sqd_regions,
			 double max_error, double *actual_error) const;

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
				double *actual_error) const;

  /** @brief Prints out the series expansion represented by this
   *         object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

  /** @brief Translate from a far field expansion to the expansion
   *         here. The translated coefficients are added up to the
   *         ones here.
   */
  void TranslateFromFarField(const InversePowDistFarFieldExpansion &se);
  
  /**
   * Translate to the given local expansion. The translated coefficients
   * are added up to the passed-in local expansion coefficients.
   */
  template<typename InversePowDistLocalExpansion>
  void TranslateToLocal(InversePowDistLocalExpansion &se, 
			int truncation_order);

};

#define INSIDE_INVERSE_POW_DIST_FARFIELD_EXPANSION_H
#include "inverse_pow_dist_farfield_expansion_impl.h"
#undef INSIDE_INVERSE_POW_DIST_FARFIELD_EXPANSION_H

#endif
