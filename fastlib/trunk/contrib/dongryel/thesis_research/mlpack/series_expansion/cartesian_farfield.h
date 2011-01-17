/** @file cartesian_farfield.h
 *
 *  This file contains a templatized class implementing $O(D^p)$ or
 *  $O(p^D)$ expansion for computing the coefficients for a far-field
 *  expansion for an arbitrary kernel function.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_CARTESIAN_FARFIELD_H
#define MLPACK_SERIES_EXPANSION_CARTESIAN_FARFIELD_H

#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "mlpack/series_expansion/cartesian_expansion_global_dev.h"

namespace mlpack {
namespace series_expansion {

/** @brief Far field expansion class for general Cartesian basis
 *         expansion.
 */
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class CartesianFarField {

  private:

    ////////// Private Member Variables //////////

    /** @brief The center of the expansion. */
    core::table::DensePoint center_;

    /** @brief The coefficients. */
    core::table::DensePoint coeffs_;

    /** @brief The order of the expansion. */
    short int order_;

  public:

    ////////// Getters/Setters //////////

    /** @brief Gets the center of expansion.
     *
     *  @return The center of expansion for the current far-field expansion.
     */
    core::table::DensePoint &get_center();

    const core::table::DensePoint &get_center() const;

    /** @brief Gets the set of far-field coefficients.
     *
     *  @return The const reference to the vector containing the
     *          far-field coefficients.
     */
    const core::table::DensePoint& get_coeffs() const;

    /** @brief Gets the approximation order.
     *
     *  @return The integer representing the current approximation order.
     */
    short int get_order() const;

    /** @brief Gets the weight sum.
     */
    double get_weight_sum() const;

    /** @brief Sets the approximation order of the far-field expansion.
     *
     *  @param new_order The desired new order of the approximation.
     */
    void set_order(short int new_order);

    /** @brief Set the center of the expansion - assumes that the center
     *         has been initialized before...
     *
     *  @param center The center of expansion whose coordinate values
     *                will be copied to the center of the given far-field
     *                expansion object.
     */
    void set_center(const core::table::DensePoint &center);

    ////////// User-level Functions //////////

    /** @brief Accumulates the far field moment represented by the given
     *         reference data into the coefficients.
     *
     *  Given the set of reference points \f$r_{j_n} \in R\f$ in the
     *  reference node \f$R\f$, this function computes
     *  \f$\sum\limits_{r_{j_n} \in R} w_{j_n} \left( \frac{r_{j_n} -
     *  R.c}{kh} \right)^{\alpha}\f$ for \f$0 \leq \alpha \leq p\f$
     *  where \f$\alpha\f$ is a \f$D\f$-dimensional multi-index and adds
     *  to the currently accumulated far-field moments: \f$ F(R)
     *  \leftarrow F(R) + \left[ \sum\limits_{r_{j_n} \in R} w_{j_n}
     *  \left( \frac{r_{j_n} - R.c}{kh} \right)^{\alpha} \right]_{0 \leq
     *  p \leq \alpha}\f$ where \f$F(R)\f$ is the current set of
     *  far-field moments for the reference node \f$R\f$. \f$k\f$ is an
     *  appropriate factor to multiply the bandwidth \f$h\f$ by; for the
     *  Gaussian kernel expansion, it is \f$\sqrt{2}\f$.
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
    template<typename KernelAuxType>
    void AccumulateCoeffs(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix &data,
      const core::table::DensePoint &weights,
      int begin, int end, int order);

    /** @brief Refine the far field moment that has been computed before
     *         up to a new order.
     */
    template<typename KernelAuxType>
    void RefineCoeffs(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix &data,
      const core::table::DensePoint &weights,
      int begin, int end, int order);

    /** @brief Evaluates the far-field coefficients at the given point.
     */
    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in,
      const core::table::DenseMatrix &data, int row_num, int order) const;

    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in, const double *x_q, int order) const;

    /** @brief Initializes the current far field expansion object with
     *         the given center.
     */
    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka, const core::table::DensePoint& center);

    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka);

    /** @brief Computes the required order for evaluating the far field
     *         expansion for any query point within the specified region
     *         for a given bound.
     */
    template<typename KernelAuxType, typename BoundType>
    int OrderForEvaluating(
      const KernelAuxType &kernel_aux_in,
      const BoundType &far_field_region,
      const BoundType &local_field_region,
      double min_dist_sqd_regions,
      double max_dist_sqd_regions,
      double max_error, double *actual_error) const;

    /** @brief Computes the required order for converting to the local
     *         expansion inside another region, so that the total error
     *         (truncation error of the far field expansion plus the
     *         conversion error) is bounded above by the given user
     *         bound.
     *
     *  @return the minimum approximation order required for the error,
     *          -1 if approximation up to the maximum order is not possible.
     */
    template<typename KernelAuxType, typename BoundType>
    int OrderForConvertingToLocal(
      const KernelAuxType &kernel_aux_in,
      const BoundType &far_field_region,
      const BoundType &local_field_region,
      double min_dist_sqd_regions,
      double max_dist_sqd_regions,
      double required_bound,
      double *actual_error) const;

    /** @brief Prints out the series expansion represented by this object.
     */
    template<typename KernelAuxType>
    void Print(
      const KernelAuxType &kernel_aux_in,
      const char *name = "", FILE *stream = stderr) const;

    /** @brief Translate from a far field expansion to the expansion
     *         here. The translated coefficients are added up to the
     *         ones here.
     */
    template<typename KernelAuxType>
    void TranslateFromFarField(
      const KernelAuxType &kernel_aux_in,
      const CartesianFarField<ExpansionType> &se);

    /** @brief Translate to the given local expansion. The translated
     *         coefficients are added up to the passed-in local
     *         expansion coefficients.
     */
    template<typename KernelAuxType, typename CartesianLocalType>
    void TranslateToLocal(
      const KernelAuxType &kernel_aux_in,
      int truncation_order, CartesianLocalType *se) const;
};

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
core::table::DensePoint &CartesianFarField<ExpansionType>::get_center() {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint &CartesianFarField <
ExpansionType >::get_center() const {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint &CartesianFarField <
ExpansionType >::get_coeffs() const {
  return coeffs_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
short int CartesianFarField<ExpansionType>::get_order() const {
  return order_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianFarField<ExpansionType>::get_weight_sum() const {
  return coeffs_[0];
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianFarField<ExpansionType>::set_order(short int new_order) {
  order_ = new_order;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianFarField<ExpansionType>::set_center(
  const core::table::DensePoint &center) {
  for(int i = 0; i < center.length(); i++) {
    center_[i] = center[i];
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
double CartesianFarField<ExpansionType>::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix& data, int row_num, int order) const {
  return EvaluateField(kernel_aux_in, data.GetColumnPtr(row_num), order);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::Init(
  const KernelAuxType &kernel_aux_in, const core::table::DensePoint& center) {

  // Copy the center.
  center_.Copy(center);
  order_ = -1;

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename TKernelAux>
void CartesianFarField<ExpansionType>::Init(const TKernelAux &kernel_aux_in) {

  // Set the center to be a zero vector.
  order_ = -1;
  center_.Init(kernel_aux_in.global().get_dimension());
  center_.SetZero();

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType, typename BoundType>
int CartesianFarField<ExpansionType>::OrderForEvaluating(
  const KernelAuxType &kernel_aux_in,
  const BoundType &far_field_region,
  const BoundType &local_field_region, double min_dist_sqd_regions,
  double max_dist_sqd_regions, double max_error, double *actual_error) const {

  return kernel_aux_in.OrderForEvaluatingFarField(
           far_field_region,
           local_field_region,
           min_dist_sqd_regions,
           max_dist_sqd_regions, max_error,
           actual_error);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType, typename BoundType>
int CartesianFarField<ExpansionType>::OrderForConvertingToLocal(
  const KernelAuxType &kernel_aux_in,
  const BoundType &far_field_region, const BoundType &local_field_region,
  double min_dist_sqd_regions, double max_dist_sqd_regions, double max_error,
  double *actual_error) const {

  return kernel_aux_in.OrderForConvertingFromFarFieldToLocal(
           far_field_region,
           local_field_region,
           min_dist_sqd_regions,
           max_dist_sqd_regions,
           max_error, actual_error);
}
}
}

#endif
