/**
 * @file inverse_pow_dist_local_expansion.h
 *
 * This file contains prototype methods for creating a local expansion
 * for an inverse power of distance functions.
 *
 * @author Dongryeol Lee (dongryel@cc.gatech.edu)
 * @bug No known bugs.
 */

#ifndef INVERSE_POW_DIST_LOCAL_EXPANSION
#define INVERSE_POW_DIST_LOCAL_EXPANSION

#include <values.h>

#include "fastlib/fastlib.h"
#include "kernel_aux.h"
#include "inverse_pow_dist_series_expansion_aux.h"

class InversePowDistFarFieldExpansion;

/** @brief The class defining a local expansion for inverse distance
 *         kernel functions.
 */
class InversePowDistLocalExpansion {

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

  OT_DEF(InversePowDistLocalExpansion) {
    OT_MY_OBJECT(inverse_power_);
    OT_MY_OBJECT(center_);
    OT_MY_OBJECT(coeffs_);
    OT_MY_OBJECT(order_);
  }

 public:
  
  ////////// Getters/Setters //////////
  
  /** @brief Get the center of expansion.
   */
  Vector* get_center() {
    return &center_;
  }

  /** @brief Retrieves the center of expansion.
   */
  const Vector* get_center() const {
    return &center_;
  }
  
  /** @brief Retrieves the set of coefficients.
   */
  ArrayList<GenMatrix<std::complex<double> > > *get_coeffs() {
    return &coeffs_;
  }

  /** @brief Retrieves the set of coefficients.
   */
  const ArrayList<GenMatrix<std::complex<double> > > *get_coeffs() const {
    return &coeffs_;
  }

  /** @brief Retrieves the approximation order.
   */
  int get_order() const { 
    return order_;
  }

  /** @brief Get the maximum possible approximation order.
   */
  int get_max_order() const {
    return sea_->get_max_order();
  }

  /** @brief Sets the approximation order.
   */
  void set_order(int new_order) {
    order_ = new_order;
  }
 
  void Accumulate(const double *v, double weight, int order);

  /**
   * Accumulates the local moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const Matrix& data, const Vector& weights,
			int begin, int end, int order);

  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateField(const Matrix& data, int row_num, int order) const;
  double EvaluateField(const double *x_q, int order) const;

  /** @brief Initializes the current local expansion object with
   *         the given center.
   */
  void Init(const Vector& center, InversePowDistSeriesExpansionAux *sea);

  /**
   * Prints out the series expansion represented by this object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

};

#define INSIDE_INVERSE_POW_DIST_LOCAL_EXPANSION_H
#include "inverse_pow_dist_local_expansion_impl.h"
#undef INSIDE_INVERSE_POW_DIST_LOCAL_EXPANSION_H

#endif
