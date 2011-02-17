/** @file reduced_set_farfield_dev.h
 *
 *  The farfield expansion using the reduced set method.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_DEV_H
#define MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_DEV_H

#include <armadillo>
#include "mlpack/series_expansion/reduced_set_farfield.h"

namespace mlpack {
namespace series_expansion {

core::table::DensePoint &ReducedSetFarField::get_center() {
  return center_;
}

const core::table::DensePoint &ReducedSetFarField::get_center() const {
  return center_;
}

const core::table::DensePoint &ReducedSetFarField::get_coeffs() const {
  return coeffs_;
}

short int ReducedSetFarField::get_order() const {
  return order_;
}

double ReducedSetFarField::get_weight_sum() const {
  return coeffs_[0];
}

void ReducedSetFarField::set_order(short int new_order) {
  order_ = new_order;
}

void ReducedSetFarField::set_center(
  const core::table::DensePoint &center) {
  for(int i = 0; i < center.length(); i++) {
    center_[i] = center[i];
  }
}

template<typename KernelAuxType>
void ReducedSetFarField::Init(
  const KernelAuxType &kernel_aux_in, const core::table::DensePoint& center) {

  // Copy the center.
  center_.Copy(center);
  order_ = -1;

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernelAux>
void ReducedSetFarField::Init(const TKernelAux &kernel_aux_in) {

  // Set the center to be a zero vector.
  order_ = -1;
  center_.Init(kernel_aux_in.global().get_dimension());
  center_.SetZero();

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename KernelAuxType>
void ReducedSetFarField::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix& data,
  const core::table::DensePoint& weights,
  int begin, int end, int order) {

}

template<typename KernelAuxType>
void ReducedSetFarField::RefineCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix &data,
  const core::table::DensePoint &weights,
  int begin, int end, int order) {

}

template<typename KernelAuxType>
double ReducedSetFarField::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &x_q) const {

  // Get the already-generated grid points. We need to map this to the
  // upward equivalent surface.
  const core::table::DenseMatrix &uniform_grid_points =
    kernel_aux_in.uniform_grid_points();

  // Take the weighted sum.
  double multipole_sum = 0.0;
  arma::vec grid_point;
  arma::vec lower_bound_upward_equivalent_alias;
  core::table::DensePointToArmaVec(
    lower_bound_upward_equivalent_, &lower_bound_upward_evauilent_alias);
  for(int i = 0; i < pseudocharges_.size(); i++) {
    arma::vec grid_point_alias;
    uniform_grid_points.MakeColumnVector(i, &grid_point_alias);
    grid_point = grid_point_alias;

    // Scale by the upward equivalent surface of the node.

    // Translate it by the lower bound.
    grid_point += lower_bound_upward_equivalent_alias;

    double kernel_value = kernel_aux_in.kernel().EvalUnnormOnSq(
                            x_q, grid_point);
    multipole_sum += pseudocharges_[i] * kernel_value;
  }
  return multipole_sum;
}

template<typename KernelAuxType>
void ReducedSetFarField::Print(
  const KernelAuxType &kernel_aux_in, const char *name, FILE *stream) const {

}

template<typename KernelAuxType>
void ReducedSetFarField::TranslateFromFarField(
  const KernelAuxType &kernel_aux_in, const ReducedSetFarField &se) {

}

template<typename KernelAuxType, typename ReducedSetLocalType>
void ReducedSetFarField::TranslateToLocal(
  const KernelAuxType &kernel_aux_in, int truncation_order,
  ReducedSetLocalType *se) const {

}
}
}

#endif
