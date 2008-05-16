/**
 * @file matrix_factorized_farfield_expansion.h
 *
 * This file contains a templatized class implementing the matrix
 * factorized far-field expansion for an arbitrary kernel function.
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef MATRIX_FACTORIZED_FARFIELD_EXPANSION
#define MATRIX_FACTORIZED_FARFIELD_EXPANSION

#include "fastlib/fastlib.h"
#include "kernel_aux.h"

template<typename TKernelAux> 
class LocalExpansion;

/** @brief Far field expansion class using matrix factorization.
 *
 *  @code
 *   // Declare a far-field expansion for the Gaussian kernel.
 *   MatrixFactorizedFarFieldExpansion<GaussianKernelAux> fe;
 *
 *  @endcode
 */
template<typename TKernelAux>
class MatrixFactorizedFarFieldExpansion {

 private:
  
  ////////// Private Member Variables //////////

  /** @brief The center of expansion.
   */
  Vector center_;

  /** @brief The projection operator.
   */
  Matrix projection_operator_;

  /** @brief The out-going representation: the pseudo-charges on the
   *         pseudo-points.
   */
  Vector outgoing_representation_;
  
  /** @brief The reference point indices that form the out-going
   *         skeleton, the pseudo-points that represent the reference
   *         point distribution.
   */
  ArrayList<index_t> outgoing_skeleton_;

  /** @brief The pointer to the auxilirary methods for the kernel
   *         (derivative, truncation error bound).
   */
  const TKernelAux *ka_;

  /** @brief The pointer to the kernel object inside kernel auxiliary
   *         object.
   */
  const typename TKernelAux::TKernel *kernel_;

  /** @brief The pointer to the precomputed constants inside kernel
   *         auxiliary object 
   */
  const typename TKernelAux::TSeriesExpansionAux *sea_;

  OT_DEF(MatrixFactorizedFarFieldExpansion) {
    OT_MY_OBJECT(center_);
    OT_MY_OBJECT(projection_operator_);
    OT_MY_OBJECT(outgoing_representation_);
    OT_MY_OBJECT(outoing_skeleton_);
  }

 public:
  
  ////////// Getters/Setters //////////
  
  /** @brief Gets the squared bandwidth value that is being used by
   *         the current far-field expansion object.
   *
   *  @return The squared bandwidth value.
   */
  double bandwidth_sq() const { return kernel_->bandwidth_sq(); }

  /** @brief Gets the center of expansion.
   *
   *  @return The center of expansion for the current far-field expansion.
   */
  Vector* get_center() { return &center_; }

  const Vector* get_center() const { return &center_; }

  /** @brief Gets the set of far-field coefficients.
   *
   *  @return The const reference to the vector containing the
   *          far-field coefficients.
   */
  const Vector& get_coeffs() const { return coeffs_; }
  
  /** @brief Set the center of the expansion - assumes that the center
   *         has been initialized before...
   *
   *  @param center The center of expansion whose coordinate values
   *                will be copied to the center of the given far-field 
   *                expansion object.
   */
  void set_center(const Vector &center) {
    
    for(index_t i = 0; i < center.length(); i++) {
      center_[i] = center[i];
    }
  }

  ////////// User-level Functions //////////
  
  /** @brief Accumulates the contribution of a single reference point as a 
   *         far-field moment.
   *
   *  @param reference_point The coordinates of the reference point.
   *  @param weight The weight of the reference point v.
   *  @param order The order up to which the far-field moments should be 
   *               accumulated up to.
   */
  void Accumulate(const Vector &reference_point, double weight, int order);

  /** @brief Accumulates the far field moment represented by the given
   *         reference data into the coefficients.
   *
   *  @param data The entire reference dataset \f$\mathcal{R}\f$.
   *  @param weights The entire reference weights \f$\mathcal{W}\f$.
   *  @param begin The beginning index of the reference points for
   *               which we would like to accumulate the moments for.
   *  @param end The upper limit on the index of the reference points for
   *             which we would like to accumulate the moments for.
   *  @param order The order up to which the far-field moments should be
   *               accumulated up to.
   */
  template<typename Tree>
  void AccumulateCoeffs
  (const Matrix& data, const Vector& weights, int begin, int end, 
   int order = -1, const Matrix *query_set = NULL,
   const ArrayList<Tree *> query_leaf_nodes = NULL);

  /** @brief Refine the far field moment that has been computed before
   *         up to a new order.
   */
  void RefineCoeffs(const Matrix& data, const Vector& weights,
		    int begin, int end, int order);
  
  /** @brief Evaluates the far-field coefficients at the given point.
   */
  double EvaluateField(const Matrix& data, int row_num, int order) const;
  double EvaluateField(const Vector &x_q, int order) const;

  /** @brief Initializes the current far field expansion object with
   *         the given center.
   */
  void Init(const Vector& center, const TKernelAux &ka);
  void Init(const TKernelAux &ka);

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

  /** @brief Computes the required order for evaluating the far field
   *         expansion for any query point within the specified region
   *         for a given bound.
   */
  template<typename TBound>
  int OrderForEvaluatingByMonteCarlo(const TBound &far_field_region,
				     const TBound &local_field_region,
				     double min_dist_sqd_regions,
				     double max_dist_sqd_regions,
				     double max_error, double *actual_error,
				     int *num_samples) const;

  /** @brief Computes the required order for converting to the local
   *         expansion inside another region, so that the total error
   *         (truncation error of the far field expansion plus the
   *         conversion error) is bounded above by the given user
   *         bound.
   *
   *  @return the minimum approximation order required for the error,
   *          -1 if approximation up to the maximum order is not possible.
   */
  template<typename TBound>
  int OrderForConvertingToLocal(const TBound &far_field_region,
				const TBound &local_field_region, 
				double min_dist_sqd_regions, 
				double max_dist_sqd_regions,
				double required_bound, 
				double *actual_error) const;

  /** @brief Prints out the series expansion represented by this object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

  /** @brief Translate from a far field expansion to the expansion
   *         here. The translated coefficients are added up to the
   *         ones here.
   */
  void TranslateFromFarField(const FarFieldExpansion &se);
  
  /** @brief Translate to the given local expansion. The translated
   *         coefficients are added up to the passed-in local
   *         expansion coefficients.
   */
  void TranslateToLocal(LocalExpansion<TKernelAux> &se, int truncation_order);

};

#define INSIDE_MATRIX_FACTORIZED_FARFIELD_EXPANSION_H
#include "matrix_factorized_farfield_expansion_impl.h"
#undef INSIDE_MATRIX_FACTORIZED_FARFIELD_EXPANSION_H

#endif
