/** @file cartesian_local.h
 *
 *  This file contains a templatized class implementing $O(D^p)$
 *  expansion for computing the coefficients for a local expansion for
 *  an arbitrary kernel function.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_CARTESIAN_LOCAL_H
#define MLPACK_SERIES_EXPANSION_CARTESIAN_LOCAL_H

#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "mlpack/series_expansion/cartesian_expansion_global_dev.h"

namespace mlpack {
namespace series_expansion {

/** @brief The general Cartesian local expansion.
 */
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class CartesianLocal {

  private:

    /** @brief The center of the expansion. */
    core::table::DensePoint center_;

    /** @brief The coefficients. */
    core::table::DensePoint coeffs_;

    /** @brief The truncation order. */
    int order_;

  public:

    /** @brief Get the center of expansion. */
    core::table::DensePoint* get_center() {
      return &center_;
    }

    /** @brief Get the center of expansion. */
    const core::table::DensePoint* get_center() const {
      return &center_;
    }

    /** @brief Get the coefficients. */
    const core::table::DensePoint& get_coeffs() const {
      return coeffs_;
    }

    /** @brief Get the approximation order. */
    int get_order() const {
      return order_;
    }

    /** @brief Get the maximum possible approximation order. */
    int get_max_order() const {
      return sea_->get_max_order();
    }

    /** @brief Set the approximation order. */
    void set_order(int new_order) {
      order_ = new_order;
    }

    /** @brief Accumulates the local moment represented by the given
     *         reference data into the coefficients.
     */
    template<typename KernelAuxType>
    void AccumulateCoeffs(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix& data,
      const core::table::DensePoint& weights,
      int begin, int end, int order);

    /** @brief Evaluates the local coefficients at the given point.
     */
    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix& data, int row_num) const;

    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in, const double *x_q) const;

    /** @brief Initializes the current local expansion object with the
     *         given center.
     */
    void Init(const core::table::DensePoint& center, const TKernelAux &ka);
    void Init(const TKernelAux &ka);

    /** @brief Computes the required order for evaluating the local
     *         expansion for any query point within the specified
     *         region for a given bound.
     */
    template<typename KernelAuxType, typename BoundType>
    int OrderForEvaluating(
      const KernelAuxType &kernel_aux_in,
      const BoundType &far_field_region,
      const BoundType &local_field_region,
      double min_dist_sqd_regions,
      double max_dist_sqd_regions,
      double max_error, double *actual_error) const;

    /** @brief Prints out the series expansion represented by this
     *         object.
     */
    template<typename KernelAuxType>
    void Print(
      const KernelAuxType &kernel_aux_in,
      const char *name = "", FILE *stream = stderr) const;

    /** @brief Translate from a far field expansion to the expansion
     *         here.The translated coefficients are added up to the
     *         ones here.
     */
    template<typename KernelAuxType, typename CartesianFarFieldType>
    void TranslateFromFarField(
      const KernelAuxType &kernel_aux_in, const CartesianFarFieldType &se);

    /** @brief Translate to the given local expansion. The translated
     *         coefficients are added up to the passed-in local
     *         expansion coefficients.
     */
    template<typename KernelAuxType>
    void TranslateToLocal(
      const KernelAuxType &kernel_aux_in, CartesianLocal<ExpansionType> *se);
};

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianLocal<ExpansionType>::Init(
  const KernelAuxType &kernel_aux_in, const core::table::DensePoint& center) {

  // Copy the center.
  center_.Copy(center);
  order_ = -1;

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianLocal<ExpansionType>::Init(const KernelAuxType &ka) {

  // Initialize the center to be zero.
  order_ = -1;
  center_.Init(sea_->get_dimension());
  center_.SetZero();

  // Initialize coefficient array.
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType, typename BoundType>
int CartesianLocal<ExpansionType>::OrderForEvaluating(
  const BoundType &far_field_region,
  const BoundType &local_field_region, double min_dist_sqd_regions,
  double max_dist_sqd_regions, double max_error, double *actual_error) const {

  return ka_->OrderForEvaluatingLocal(far_field_region, local_field_region,
                                      min_dist_sqd_regions,
                                      max_dist_sqd_regions, max_error,
                                      actual_error);
}
}
}

#endif
