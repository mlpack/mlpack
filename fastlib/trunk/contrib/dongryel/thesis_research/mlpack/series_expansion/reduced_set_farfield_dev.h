/** @file reduced_set_farfield_dev.h
 *
 *  The farfield expansion using the reduced set method.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_DEV_H
#define MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_DEV_H

#include <armadillo>
#include "mlpack/series_expansion/dense_matrix_inverse.h"
#include "mlpack/series_expansion/reduced_set_farfield.h"

namespace mlpack {
namespace series_expansion {

ReducedSetFarField::ReducedSetFarField() {
  current_kernel_matrix_ = NULL;
  current_kernel_matrix_inverse_ = NULL;
}

ReducedSetFarField::~ReducedSetFarField() {
  if(current_kernel_matrix_ != NULL) {
    delete current_kernel_matrix_;
  }
  if(current_kernel_matrix_inverse_ != NULL) {
    delete current_kernel_matrix_inverse_;
  }
}

bool ReducedSetFarField::in_dictionary(int training_point_index) const {
  return in_dictionary_[training_point_index];
}

void ReducedSetFarField::UpdateDictionary_(
  int new_point_index,
  const arma::vec &new_column_vector,
  double self_value,
  double projection_error,
  const arma::vec &inverse_times_column_vector) {

  // Add the point to the dictionary.
  point_indices_in_dictionary_.push_back(new_point_index);
  in_dictionary_[ new_point_index ] = true;
  training_index_to_dictionary_position_[ new_point_index ] =
    point_indices_in_dictionary_.size() - 1;

  // Update the kernel matrix.
  core::table::DenseMatrix *new_kernel_matrix = new core::table::DenseMatrix();
  new_kernel_matrix->Init(
    current_kernel_matrix_->n_rows() + 1, current_kernel_matrix_->n_cols() + 1);

  for(int j = 0; j < current_kernel_matrix_->n_cols(); j++) {
    for(int i = 0; i < current_kernel_matrix_->n_rows(); i++) {
      new_kernel_matrix->set(i, j, current_kernel_matrix_->get(i, j));
    }
  }
  for(int j = 0; j < current_kernel_matrix_->n_cols(); j++) {
    new_kernel_matrix->set(
      j, current_kernel_matrix_->n_cols(), new_column_vector[j]);
    new_kernel_matrix->set(
      current_kernel_matrix_->n_rows(), j, new_column_vector[j]);
  }

  // Store the self value.
  new_kernel_matrix->set(
    current_kernel_matrix_->n_rows(),
    current_kernel_matrix_->n_cols(),
    self_value);
  delete current_kernel_matrix_;
  current_kernel_matrix_ = new_kernel_matrix;

  // Update the kernel matrix inverse.
  core::table::DenseMatrix *new_kernel_matrix_inverse =
    mlpack::series_expansion::DenseMatrixInverse::Update(
      *current_kernel_matrix_inverse_,
      inverse_times_column_vector,
      projection_error);
  delete current_kernel_matrix_inverse_;
  current_kernel_matrix_inverse_ = new_kernel_matrix_inverse;
}

const core::table::DenseMatrix *ReducedSetFarField::current_kernel_matrix()
const {
  return current_kernel_matrix_;
}

const core::table::DenseMatrix *ReducedSetFarField::current_kernel_matrix_inverse() const {
  return current_kernel_matrix_inverse_;
}

core::table::DenseMatrix *ReducedSetFarField::current_kernel_matrix() {
  return current_kernel_matrix_;
}

core::table::DenseMatrix *ReducedSetFarField::current_kernel_matrix_inverse() {
  return current_kernel_matrix_inverse_;
}

void ReducedSetFarField::AddBasis(
  int new_point_index,
  const arma::vec &new_column_vector_in,
  double self_value) {

  static const double adding_threshold_ = 1e-5;

  if(new_column_vector_in.n_elem > 0) {

    // Compute the matrix-vector product.
    arma::mat current_kernel_matrix_inverse_alias;
    core::table::DenseMatrixToArmaMat(
      *current_kernel_matrix_inverse_, &current_kernel_matrix_inverse_alias);
    arma::vec inverse_times_column_vector =
      current_kernel_matrix_inverse_alias * new_column_vector_in;

    // Compute the projection error.
    double projection_error =
      self_value - arma::dot(new_column_vector_in, inverse_times_column_vector);

    // If the projection error is above the threshold, add it to the
    // dictionary.
    if(projection_error > adding_threshold_) {
      UpdateDictionary_(
        new_point_index,
        new_column_vector_in,
        self_value,
        projection_error,
        inverse_times_column_vector);
    }
  }
  else {
    current_kernel_matrix_ = new core::table::DenseMatrix();
    current_kernel_matrix_->Init(1, 1);
    current_kernel_matrix_->set(0, 0, self_value);
    current_kernel_matrix_inverse_ = new core::table::DenseMatrix();
    current_kernel_matrix_inverse_->Init(1, 1);
    current_kernel_matrix_inverse_->set(0, 0, self_value);
  }
}

template<typename TreeIteratorType>
void ReducedSetFarField::Init(const TreeIteratorType &it) {

  // Allocate the boolean flag for the presence of each training
  // point in the dictionary.
  in_dictionary_.resize(it.count());
  training_index_to_dictionary_position_.resize(it.count());
  point_indices_in_dictionary_.resize(0);

  for(int i = 0; i < it.count(); i++) {
    in_dictionary_[i] = false;
    training_index_to_dictionary_position_[i] = -1;
  }
}

int ReducedSetFarField::training_index_to_dictionary_position(
  int training_index) const {
  return training_index_to_dictionary_position_[training_index];
}

int ReducedSetFarField::point_indices_in_dictionary(
  int nth_dictionary_point_index) const {
  return point_indices_in_dictionary_[nth_dictionary_point_index];
}

const std::vector<bool> &ReducedSetFarField::in_dictionary() const {
  return in_dictionary_;
}

const std::vector<int> &ReducedSetFarField::point_indices_in_dictionary()
const {
  return point_indices_in_dictionary_;
}

const std::vector<int> &ReducedSetFarField::
training_index_to_dictionary_position() const {
  return training_index_to_dictionary_position_;
}

template <
typename MetricType, typename KernelAuxType, typename TreeIteratorType >
void ReducedSetFarField::FillKernelValues_(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &candidate,
  TreeIteratorType &it,
  arma::vec *kernel_values_out,
  double *self_value) const {

  // Resize the kernel value vector.
  kernel_values_out->set_size(point_indices_in_dictionary_.size());

  // First the self kernel value.
  *self_value = kernel_aux_in.kernel().EvalUnnormOnSq(0.0);

  for(unsigned int i = 0; i < point_indices_in_dictionary_.size(); i++) {
    core::table::DensePoint dictionary_point;
    it.get(point_indices_in_dictionary_[i], &dictionary_point);
    double squared_distance =
      metric_in.DistanceSq(candidate, dictionary_point);
    (*kernel_values_out)[i] =
      kernel_aux_in.kernel().EvalUnnormOnSq(squared_distance);
  }
}

template<typename MetricType, typename KernelAuxType, typename TreeIteratorType>
void ReducedSetFarField::AccumulateCoeffs(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  TreeIteratorType &it) {

  // Loop through each point and build the dictionary.
  it.Reset();
  arma::vec new_column_vector_in;
  while(it.HasNext()) {
    core::table::DensePoint point;
    it.Next(&point);

    // The DFS index shifted so that the begin index is 0.
    int current_index = it.current_index() - it.begin();

    // Fill out the kernel values, and do the self-computation.
    double self_value;
    FillKernelValues_(
      metric_in, kernel_aux_in, point, it, &new_column_vector_in, &self_value);
    AddBasis(
      current_index, new_column_vector_in, self_value);
  }
}

template<typename KernelAuxType>
double ReducedSetFarField::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &x_q) const {
  return 0;
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
