/** @file dictionary_dev.h
 *
 *  @brief A generic dictionary for subset of regressor methods.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_GP_REGRESSION_DICTIONARY_DEV_H
#define ML_GP_REGRESSION_DICTIONARY_DEV_H

#include <deque>
#include <vector>
#include "fastlib/fastlib.h"
#include "dense_matrix_inverse.h"
#include "dictionary.h"

namespace ml {
void Dictionary::UpdateDictionary_(
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

  for (int j = 0; j < current_kernel_matrix_->n_cols(); j++) {
    for (int i = 0; i < current_kernel_matrix_->n_rows(); i++) {
      new_kernel_matrix->set(i, j, current_kernel_matrix_->get(i, j));
    }
  }
  for (int j = 0; j < current_kernel_matrix_->n_cols(); j++) {
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
    ml::DenseKernelMatrixInverse::Update(
      *current_kernel_matrix_inverse_,
      inverse_times_column_vector,
      projection_error);
  delete current_kernel_matrix_inverse_;
  current_kernel_matrix_inverse_ = new_kernel_matrix_inverse;
}

Matrix *Dictionary::current_kernel_matrix() {
  return current_kernel_matrix_;
}

Matrix *Dictionary::current_kernel_matrix_inverse() {
  return current_kernel_matrix_inverse_;
}

int Dictionary::size() const {
  return point_indices_in_dictionary_.size();
}

void Dictionary::AddBasis(
  const int &iteration_number,
  const Vector &new_column_vector,
  double self_value) {

  // The threshold for determining whether to add a given new
  // point to the dictionary or not.
  const double adding_threshold = 1e-3;

  // The new point to consider for adding.
  int new_point_index = random_permutation_[iteration_number];

  // The vector for storing kernel values.
  Vector new_column_vector;
  new_column_vector.Init(point_indices_in_dictionary_.size());

  // Compute the matrix-vector product.
  Vector inverse_times_column_vector;
  la::MulInit(
    *current_kernel_matrix_inverse_,
    new_column_vector,
    &inverse_times_column_vector);

  // Compute the projection error.
  double projection_error =
    self_value -
    la::Dot(new_column_vector, inverse_times_kernel_vector);

  // If the projection error is above the threshold, add it to the
  // dictionary.
  if (projection_error > adding_threshold) {
    UpdateDictionary_(
      new_point_index,
      new_column_vector,
      self_value,
      projection_error,
      inverse_times_column_vector);
  }
}

void Dictionary::Init(const Matrix *table_in) {

  table_ = table_in;

  // Allocate the boolean flag for the presence of each training
  // point in the dictionary.
  in_dictionary_.resize(table_in->n_cols());
  training_index_to_dictionary_position_.resize(table_in->n_cols());

  // Generate a random permutation and initialize the inital
  // dictionary which consists of the first random point.
  random_permutation_.resize(table_in->n_cols());
  for (int i = 0; i < table_in->n_cols(); i++) {
    random_permutation_[i] = i;
    in_dictionary_[i] = false;
    training_index_to_dictionary_position_[i] = -1;
  }
}
};

#endif
