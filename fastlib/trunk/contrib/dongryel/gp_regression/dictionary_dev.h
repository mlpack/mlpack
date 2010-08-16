/** @file dictionary_dev.h
 *
 *  @brief A generic dictionary for subset of regressor methods.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_GP_REGRESSION_DICTIONARY_DEV_H
#define MLPACK_GP_REGRESSION_DICTIONARY_DEV_H

#include <deque>
#include <vector>
#include "fastlib/fastlib.h"
#include "dense_matrix_inverse.h"
#include "dictionary.h"

namespace ml {

bool Dictionary::in_dictionary(int training_point_index) const {
  return in_dictionary_[training_point_index];
}

Dictionary::Dictionary() {
  table_ = NULL;
  current_kernel_matrix_ = NULL;
  current_kernel_matrix_inverse_ = NULL;
  adding_threshold_ = 0;
}

Dictionary::~Dictionary() {
  if(current_kernel_matrix_ != NULL) {
    delete current_kernel_matrix_;
  }
  if(current_kernel_matrix_inverse_ != NULL) {
    delete current_kernel_matrix_inverse_;
  }
}

Dictionary::Dictionary(const Dictionary &dictionary_in) {
  table_ = dictionary_in.table();
  random_permutation_ = dictionary_in.random_permutation();
  in_dictionary_ = dictionary_in.in_dictionary();
  point_indices_in_dictionary_ =
    dictionary_in.point_indices_in_dictionary();
  training_index_to_dictionary_position_ =
    dictionary_in.training_index_to_dictionary_position();
  current_kernel_matrix_ = new Matrix();
  current_kernel_matrix_->Copy(* dictionary_in.current_kernel_matrix());
  current_kernel_matrix_inverse_ = new Matrix();
  current_kernel_matrix_inverse_->Copy(
    * dictionary_in.current_kernel_matrix_inverse());
  adding_threshold_ = dictionary_in.adding_threshold();
}

void Dictionary::inactive_indices(
  std::vector<int> *inactive_indices_out) const {

  // Scan the in_dictionary list and build the inactive index set.
  inactive_indices_out->resize(0);
  for(int i = 0; i < in_dictionary_.size(); i++) {
    if(in_dictionary_[i] == false) {
      inactive_indices_out->push_back(i);
    }
  }
}

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

double Dictionary::adding_threshold() const {
  return adding_threshold_;
}

void Dictionary::set_adding_threshold(double adding_threshold_in) {
  adding_threshold_ = adding_threshold_in;
}

int Dictionary::size() const {
  return point_indices_in_dictionary_.size();
}

void Dictionary::AddBasis(
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
      la::Dot(new_column_vector, inverse_times_kernel_vector);

    // If the projection error is above the threshold, add it to the
    // dictionary.
    if(projection_error > adding_threshold_) {
      UpdateDictionary_(
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

void Dictionary::Init(const Matrix *table_in) {

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

int Dictionary::position_to_training_index_map(int position) const {
  return random_permutation_[ position ];
}

int Dictionary::training_index_to_dictionary_position(int training_index) const {
  return training_index_to_dictionary_position_[training_index];
}

int Dictionary::point_indices_in_dictionary(int nth_dictionary_point_index) const {
  return point_indices_in_dictionary_[nth_dictionary_point_index];
}

const Matrix *Dictionary::table() const {
  return table_;
}

const std::vector<int> &Dictionary::random_permutation() const {
  return random_permutation_;
}

const std::deque<bool> &Dictionary::in_dictionary() const {
  return in_dictionary_;
}

const std::vector<int> &Dictionary::point_indices_in_dictionary() const {
  return point_indices_in_dictionary_;
}

const std::vector<int> &Dictionary::training_index_to_dictionary_position() const {
  return training_index_to_dictionary_position_;
}

const Matrix *Dictionary::current_kernel_matrix() const {
  return current_kernel_matrix_;
}

const Matrix *Dictionary::current_kernel_matrix_inverse() const {
  return current_kernel_matrix_inverse_;
}

int Dictionary::size() const {
  return point_indices_in_dictionary_.size();
}
};

#endif
