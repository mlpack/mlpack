/** @file cartesian_local.h
 *
 *  This file contains a templatized class implementing $O(D^p)$ or
 *  $O(p^D)$ expansion for computing the coefficients for a local
 *  expansion for an arbitrary kernel function.
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
    core::table::DensePoint &get_center();

    /** @brief Get the center of expansion. */
    const core::table::DensePoint &get_center() const;

    /** @brief Get the coefficients. */
    const core::table::DensePoint& get_coeffs() const;

    /** @brief Get the approximation order. */
    int get_order() const;

    /** @brief Set the approximation order. */
    void set_order(int new_order);

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
      const core::table::DensePoint &x_q) const;

    /** @brief Initializes the current local expansion object with the
     *         given center.
     */
    template<typename KernelAuxType>
    void Init(
      const KernelAuxType &kernel_aux_in,
      const core::table::DensePoint& center);

    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka);

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
      const KernelAuxType &kernel_aux_in,
      CartesianLocal<ExpansionType> *se) const;
};

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
core::table::DensePoint &CartesianLocal<ExpansionType>::get_center() {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint &CartesianLocal <
ExpansionType >::get_center() const {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint& CartesianLocal <
ExpansionType >::get_coeffs() const {
  return coeffs_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianLocal<ExpansionType>::get_order() const {
  return order_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianLocal<ExpansionType>::set_order(int new_order) {
  order_ = new_order;
}

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
void CartesianLocal<ExpansionType>::Init(const KernelAuxType &kernel_aux_in) {

  // Initialize the center to be zero.
  order_ = -1;
  center_.Init(kernel_aux_in.global().get_dimension());
  center_.SetZero();

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}
}
}

#endif
