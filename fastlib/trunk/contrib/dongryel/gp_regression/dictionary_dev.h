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
  const Vector &temp_kernel_vector,
  double self_kernel_value,
  double projection_error,
  const Vector &inverse_times_kernel_vector) {

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
      j, current_kernel_matrix_->n_cols(), temp_kernel_vector[j]);
    new_kernel_matrix->set(
      current_kernel_matrix_->n_rows(), j, temp_kernel_vector[j]);
  }

  // Store the self-kernel value.
  new_kernel_matrix->set(
    current_kernel_matrix_->n_rows(),
    current_kernel_matrix_->n_cols(),
    self_kernel_value);

  delete current_kernel_matrix_;
  current_kernel_matrix_ = new_kernel_matrix;

  // Update the kernel matrix inverse.
  Matrix *new_kernel_matrix_inverse =
    fl::ml::DenseKernelMatrixInverse::Update(
      *current_kernel_matrix_inverse_,
      inverse_times_kernel_vector,
      projection_error);
  delete current_kernel_matrix_inverse_;
  current_kernel_matrix_inverse_ = new_kernel_matrix_inverse;
}
};

#endif
