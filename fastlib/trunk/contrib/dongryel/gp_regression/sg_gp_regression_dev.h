/** @file sg_gp_regression_dev.h
 *
 *  @brief An implementation of "Sparse Greedy Gaussian Process
 *         regression" by Smola et al.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_GP_REGRESSION_SG_GP_REGRESSION_DEV_H
#define ML_GP_REGRESSION_SG_GP_REGRESSION_DEV_H

#include "sg_gp_regression.h"

namespace ml {
namespace gp_regression {

void SparseGreedyGprModel::QuadraticObjective_(
  const ml::Dictionary &dictionary_in) const {


}

void SparseGreedyGprModel::FillSquaredKernelMatrix_(
  int candidate_index,
  const Vector &kernel_values,
  double noise_level_in,
  Vector *new_column_vector_out,
  double *new_self_value_out) const {

  new_column_vector_out->Init(dictionary_for_error_->size());
  const std::vector<int> &point_indices_in_dictionary =
    dictionary_for_error_.point_indices_in_dictionary();

  for (int i = 0; i < point_indices_in_dictionary.size(); i++) {
    int basis_index = point_indices_in_dictionary[i];
    const std::vector<double> &cached_kernel_values =
      kernel_matrix_columns_[basis_index];

    // Take the dot product.
    double dot_product = 0;
    for (int j = 0; j < kernel_values.length(); j++) {
      if (j == candidate_index) {
        dot_product += cached_kernel_values[j] *
                       (kernel_values[j] + noise_level_in);
      }
      else {
        dot_product += cached_kernel_values[j] * kernel_values[j];
      }
    }
    (*new_column_vector_out)[i] = dot_product;
  }
  *new_self_value_out = la::Dot(kernel_values, kernel_values) +
                        noise_level_in * kernel_values[candidate_index];
}

void SparseGreedyGprModel::FillKernelMatrix_(
  int candidate_index,
  const Vector &kernel_values,
  double noise_level_in,
  Vector *new_column_vector_out,
  double *new_self_value_out) const {

  new_column_vector_out->Init(dictionary_for_error_->size());
  const std::vector<int> &point_indices_in_dictionary =
    dictionary_for_error_.point_indices_in_dictionary();

  // Simply the necessary kernel values. For the self value, add the
  // noise.
  for (int i = 0; i < new_column_vector_out->length(); i++) {
    (*new_column_vector_out)[i] =
      kernel_values[ point_indices_in_dictionary[i] ];
  }
  *new_self_value_out = kernel_values[candidate_index] + noise_level_in;
}

template<typename CovarianceType>
void SparseGreedyGprModel::ComputeKernelValues_(
  const CovarianceType &covariance_in,
  int candidate_index,
  std::vector<double> *kernel_values_out) const {

  Vector candidate_point;
  dataset_->MakeColumnVector(candidate_index, &candidate_point);

  // Fill out the kernel values sequentially.
  for (int i = 0; i < dataset_->n_cols(); i++) {
    Vector point;
    dataset_->MakeColumnVector(i, &point);
    (*kernel_values_out)[i] = covariance_in.Dot(
                                candidate_point, point, candidate_index == i);
  }
}

template<typename CovarianceType>
void SparseGreedyGprModel::AddOptimalPoint(
  const CovarianceType &covariance_in,
  const std::vector<int> &candidate_indices,
  bool for_coeffs) {

  // The kernel values.
  std::vector<double> kernel_values(dataset_->n_cols());

  // The optimal point information.
  int optimal_point_index = -1;
  double optimum_value = std::numeric_limits<double>::max();

  // Loop over candidates and decide to add the optimal.
  for (int i = 0; i < candidate_indices.size(); i++) {

    // Make a copy of the dictionaries.
    Dictionary dictionary_copy;
    if (for_coeffs) {
      dictionary_copy = dictionary_;
    }
    else {
      dictionary_copy = dictionary_for_error_;
    }

    // Candidate index for which the kernel values have to be computed.
    int candidate_index = candidate_indices[i];
    ComputeKernelValues_(covariance_in, candidate_index, &kernel_values);

    // The new column vector to be appended and the new self value.
    Vector new_column_vector;
    double new_self_value;

    // Compute the additional quantities to be appended to grow the
    // matrix and update the dictionary.
    if (for_coeffs) {
      FillSquaredKernelMatrix_(
        candidate_index, kernel_values, &new_column_vector, &new_self_value);
      dictionary_.AddBasis(
        candidate_index, new_column_vector, new_self_value);

      // Compute the objective function value for the coefficients.
      QuadraticObjective_(dictionary_);
    }
    else {
      FillKernelMatrix_(
        candidate_index, kernel_values, &new_column_vector, &new_self_value);
      dictionary_for_error_.AddBasis(
        candidate_index, new_column_vector, new_self_value);

      // Compute the objective function value for the error bar.
      QuadraticObjective_(dictionary_for_error_);
    }
  }
}

void SparseGreedyGprModel::Init(
  const Matrix *dataset_in, const Vector *targets_in) {

  dataset_ = dataset_in;
  targets_ = targets_in;
  frobenius_norm_targets_ = la::Dot(*targets_in, *targets_in);
}

SparseGreedyGprModel::SparseGreedyGprModel() {
  dataset_ = NULL;
  targets_ = NULL;
}

SparseGreedyGpr::SparseGreedyGpr() {
  dataset_ = NULL;
  targets_ = NULL;
}

void SparseGreedyGpr::ChooseRandomSubset_(
  const std::vector<int> &inactive_set,
  int subset_size,
  std::vector<int> *subset_out) const {

  // Copy
  subset_out->resize(inactive_set.size());
  for (int i = 0; i < inactive_set.size(); i++) {
    (*subset_out)[i] = inactive_set[i];
  }

  // Then shuffle, and truncate.
  if (inactive_set.size() > subset_size) {
    for (int i = subset_out->size() - 1; i >= 1; i--) {

      // Pick a random index between 0 and i, inclusive.
      int random_index = math::RandInt(0, i);
      std::swap((*subset_out)[i], (*subset_out)[random_index]);
    }
    subset_out->resize(subset_size);
  }
}

void SparseGreedyGpr::InitInactiveSet_(
  std::vector<int> *inactive_indices_out) const {

  inactive_indices_out->resize(dataset_->n_cols());
  for (int i = 0; i < inactive_indices_out->size(); i++) {
    (*inactive_indices_out)[i] = i;
  }
}

void SparseGreedyGpr::Init(
  const Matrix &dataset_in, const Vector &targets_in) {

  dataset_ = &dataset_in;
  targets_ = &targets_in;
}

template<typename CovarianceType>
void SparseGreedyGpr::Compute(
  const CovarianceType &covariance_in,
  double noise_level_in,
  double precision_in,
  SparseGreedyGprModel *model_out) {

  // Initialize the model.
  model_out->Init(dataset_, targets_);

  // Initialize the initial index sets to choose from (inactive
  // sets).
  std::vector<int> inactive_indices(dataset_->n_cols());
  std::vector<int> inactive_indices_for_error(dataset-- > n_cols());
  for (int i = 0; i < dataset_->n_cols(); i++) {
    inactive_indices[i] = i;
    inactive_indices_for_error[i] = i;
  }

  do {

    // The candidate sets in the current iteration.
    std::vector<int> candidate_indices;
    std::vector<int> candidate_indices_for_error;

    // Choose a random subset from the inactive point set.
    ChooseRandomSubset_(
      inactive_indices, random_subset_size_, &candidate_indices);
    ChooseRandomSubset_(
      inactive_indices_for_error, random_subset_size_,
      &candidate_indices_for_error);

    // Choose a random optimal point for both sets.
    model_out->AddOptimalPoint(
      covariance_in, noise_level_in, candidate_indices, false);
    model_out->AddOptimalPoint(
      covariance_in, noise_level_in, candidate_indices_for_error, true);

    // Update the list of inactive indices.
    dictionary_.inactive_indices(&inactive_indices);
    dictionary_for_error_.inactive_indices(&inactive_indices_for_error_);
  }
  while (Done_());
}
};
};

#endif
