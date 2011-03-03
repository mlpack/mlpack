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

bool ReducedSetFarField::in_dictionary(int training_point_index) const {
  return in_dictionary_[training_point_index];
}

void ReducedSetFarField::inactive_indices(
  std::vector<int> *inactive_indices_out) const {

  // Scan the in_dictionary list and build the inactive index set.
  inactive_indices_out->resize(0);
  for(unsigned int i = 0; i < in_dictionary_.size(); i++) {
    if(in_dictionary_[i] == false) {
      inactive_indices_out->push_back(i);
    }
  }
}

void ReducedSetFarField::UpdateReducedSetFarField_(
  int new_point_index,
  const Vector &new_column_vector,
  double self_value,
  double projection_error,
  const Vector &inverse_times_column_vector) {

  // Add the point to the dictionary.
  point_indices_in_dictionary_.push_back(new_point_index);
  in_dictionary_[ new_point_index ] = true;
  training_index_to_dictionary_position_[ new_point_index ] =
    point_indices_in_dictionary_.size() - 1;

  // Update the kernel matrix.
  Matrix *new_kernel_matrix = new Matrix();
  new_kernel_matrix->Init(
    current_kernel_matrix_->n_rows() + 1,
    current_kernel_matrix_->n_cols() + 1);

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
  Matrix *new_kernel_matrix_inverse =
    ml::DenseMatrixInverse::Update(
      *current_kernel_matrix_inverse_,
      inverse_times_column_vector,
      projection_error);
  delete current_kernel_matrix_inverse_;
  current_kernel_matrix_inverse_ = new_kernel_matrix_inverse;
}

const Matrix *ReducedSetFarField::current_kernel_matrix() const {
  return current_kernel_matrix_;
}

const Matrix *ReducedSetFarField::current_kernel_matrix_inverse() const {
  return current_kernel_matrix_inverse_;
}

Matrix *ReducedSetFarField::current_kernel_matrix() {
  return current_kernel_matrix_;
}

Matrix *ReducedSetFarField::current_kernel_matrix_inverse() {
  return current_kernel_matrix_inverse_;
}

double ReducedSetFarField::adding_threshold() const {
  return adding_threshold_;
}

void ReducedSetFarField::set_adding_threshold(double adding_threshold_in) {
  adding_threshold_ = adding_threshold_in;
}

int ReducedSetFarField::size() const {
  return point_indices_in_dictionary_.size();
}

void ReducedSetFarField::AddBasis(
  int new_point_index,
  const std::vector<double> &new_column_vector_in,
  double self_value) {

  if(new_column_vector_in.size() > 0) {

    Vector new_column_vector;
    new_column_vector.Init(new_column_vector_in.size());
    for(int i = 0; i < new_column_vector.length(); i++) {
      new_column_vector[i] = new_column_vector_in[i];
    }

    // Compute the matrix-vector product.
    Vector inverse_times_column_vector;
    la::MulInit(
      *current_kernel_matrix_inverse_,
      new_column_vector,
      &inverse_times_column_vector);

    // Compute the projection error.
    double projection_error =
      self_value -
      la::Dot(new_column_vector, inverse_times_column_vector);

    // If the projection error is above the threshold, add it to the
    // dictionary.
    if(projection_error > adding_threshold_) {
      UpdateReducedSetFarField_(
        new_point_index,
        new_column_vector,
        self_value,
        projection_error,
        inverse_times_column_vector);
    }
  }
  else {
    current_kernel_matrix_ = new Matrix();
    current_kernel_matrix_->Init(1, 1);
    current_kernel_matrix_->set(0, 0, self_value);
    current_kernel_matrix_inverse_ = new Matrix();
    current_kernel_matrix_inverse_->Init(1, 1);
    current_kernel_matrix_inverse_->set(0, 0, self_value);
  }
}

void ReducedSetFarField::Init(const Matrix *table_in) {

  table_ = table_in;

  // Allocate the boolean flag for the presence of each training
  // point in the dictionary.
  in_dictionary_.resize(table_in->n_cols());
  training_index_to_dictionary_position_.resize(table_in->n_cols());

  // Generate a random permutation and initialize the inital
  // dictionary which consists of the first random point.
  random_permutation_.resize(table_in->n_cols());
  for(int i = 0; i < table_in->n_cols(); i++) {
    random_permutation_[i] = i;
    in_dictionary_[i] = false;
    training_index_to_dictionary_position_[i] = -1;
  }
}

int ReducedSetFarField::position_to_training_index_map(int position) const {
  return random_permutation_[ position ];
}

int ReducedSetFarField::training_index_to_dictionary_position(
  int training_index) const {
  return training_index_to_dictionary_position_[training_index];
}

int ReducedSetFarField::point_indices_in_dictionary(
  int nth_dictionary_point_index) const {
  return point_indices_in_dictionary_[nth_dictionary_point_index];
}

const std::vector<int> &ReducedSetFarField::random_permutation() const {
  return random_permutation_;
}

const std::deque<bool> &ReducedSetFarField::in_dictionary() const {
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

template<typename TKernelAux>
void ReducedSetFarField::Init(const TKernelAux &kernel_aux_in) {
}

template<typename KernelAuxType, typename TreeIteratorType>
void ReducedSetFarField::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint& weights,
  TreeIteratorType &it) {

}

template<typename KernelAuxType>
double ReducedSetFarField::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &x_q) const {

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
