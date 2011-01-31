/**
 * @file mult_farfield_expansion.h
 *
 * This file contains a templatized class implementing $O(p^D)$
 * expansion for computing the coefficients for a far-field expansion
 * for a arbitrary multiplicative kernel function.
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef MULT_FARFIELD_EXPANSION
#define MULT_FARFIELD_EXPANSION

#include <fastlib/fastlib.h>
#include "kernel_aux.h"
#include "mult_series_expansion_aux.h"

template<typename TKernelAux> 
class MultLocalExpansion;

/**
 * Far field expansion class
 */
template<typename TKernelAux>
class MultFarFieldExpansion {
  
 private:

  /** @brief The center of the expansion. */
  arma::vec center_;
  
  /** @brief The coefficients. */
  arma::vec coeffs_;

  /** @brief The order of approximation. */
  int order_;
  
  /** auxilirary methods for the kernel (derivative, truncation error bound) */
  const TKernelAux *ka_;

  /** pointer to the kernel object inside kernel auxiliary object */
  const typename TKernelAux::TKernel *kernel_;

  /** pointer to the precomputed constants inside kernel auxiliary object */
  const typename TKernelAux::TSeriesExpansionAux *sea_;

 public:
  
  // getters and setters
  
  /** Get the coefficients */
  double bandwidth_sq() const { return kernel_->bandwidth_sq(); }
  
  /** Get the center of expansion */
  arma::vec& get_center() { return center_; }

  const arma::vec& get_center() const { return center_; }

  /** Get the coefficients */
  arma::vec& get_coeffs() { return coeffs_; }
  const arma::vec& get_coeffs() const { return coeffs_; }
  
  /** Get the approximation order */
  int get_order() const { return order_; }
  
  /** Get the maximum possible approximation order */
  int get_max_order() const { return sea_->get_max_order(); }

  /** @brief Gets the weight sum.
   */
  double get_weight_sum() const { return coeffs_[0]; }

  /** Set the approximation order */
  void set_order(int new_order) { order_ = new_order; }
  
  /** 
   * Set the center of the expansion - assumes that the center has been
   * initialized before...
   */
  void set_center(const arma::vec& center) {
    
    for(index_t i = 0; i < center.n_elem; i++) {
      center_[i] = center[i];
    }
  }

  // interesting functions...
  
  /**
   * Accumulates the far field moment represented by the given reference
   * data into the coefficients
   */
  void AccumulateCoeffs(const arma::mat& data, const arma::vec& weights,
			int begin, int end, int order);

  /**
   * Refine the far field moment that has been computed before up to
   * a new order.
   */
  void RefineCoeffs(const arma::mat& data, const arma::vec& weights,
		    int begin, int end, int order);
  
  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateField(const arma::mat& data, int row_num, int order) const;
  double EvaluateField(const double *x_q, int order) const;

  /**
   * Evaluates the two-way convolution mixed with exhaustive computations
   * with two other far field expansions
   */
  double MixField(const arma::mat& data, int node1_begin, int node1_end, 
		  int node2_begin, int node2_end,
		  const MultFarFieldExpansion& fe2,
		  const MultFarFieldExpansion& fe3,
		  int order2, int order3) const;

  double ConvolveField(const MultFarFieldExpansion& fe, int order) const;

  /**
   * Evaluates the three-way convolution with two other far field
   * expansions
   */
  double ConvolveField(const MultFarFieldExpansion& fe2,
		       const MultFarFieldExpansion& fe3,
		       int order1, int order2, int order3) const;
  
  /**
   * Initializes the current far field expansion object with the given
   * center.
   */
  void Init(const arma::vec& center, const TKernelAux& ka);
  void Init(const TKernelAux& ka);

  /** @brief Computes the required order for evaluating the far field
   *         expansion for any query point within the specified region
   *         for a given bound.
   */
  template<typename TBound>
  int OrderForEvaluating(const TBound& far_field_region,
			 const TBound& local_field_region,
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
  int OrderForConvertingToLocal(const TBound& far_field_region,
				const TBound& local_field_region, 
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
  void TranslateFromFarField(const MultFarFieldExpansion& se);
  
  /**
   * Translate to the given local expansion. The translated coefficients
   * are added up to the passed-in local expansion coefficients.
   */
  void TranslateToLocal(MultLocalExpansion<TKernelAux>& se, 
			int truncation_order);

};

#define INSIDE_MULT_FARFIELD_EXPANSION_H
#include "mult_farfield_expansion_impl.h"
#undef INSIDE_MULT_FARFIELD_EXPANSION_H

#endif
