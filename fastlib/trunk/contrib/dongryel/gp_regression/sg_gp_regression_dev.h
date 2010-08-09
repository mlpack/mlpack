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

void SparseGreedyGprModel::GrowMatrix_(
  std::vector< std::vector<double> > &matrix) {

  // Grow the pre-existing columns by one.
  for (int i = 0; i < matrix.size(); i++) {
    matrix[i].resize(matrix[i].size() + 1);
  }

  // Increment the number of columns, and allocate the number of rows
  // for the last column.
  matrix.resize(matrix.size() + 1);
  matrix[ matrix.size() - 1 ].resize(matrix.size());
}

template<typename CovarianceType>
void SparseGreedyGprModel::ComputeKernelValues_(
  const CovarianceType &covariance_in,
  int candidate_index,
  std::vector<double> *kernel_values_out) const {

  Vector candidate_point;
  dataset_->MakeColumnVector(candidate_index, &candidate_point);

  for (int i = 0; i < dataset_->n_cols(); i++) {
    Vector point;
    dataset_->MakeColumnVector(i, &point);
    (*kernel_values_out)[i] = covariance_in.Covariance(
                                candidate_point, point, candidate_index == i);
  }
}

template<typename CovarianceType>
void SparseGreedyGprModel::AddOptimalPoint(
  const CovarianceType &covariance_in,
  double noise_level_in,
  const std::vector<int> &candidate_indices,
  bool for_coeffs) {

  // Grow the kernel matrices accordingly by one.
  if (for_coeffs) {
    GrowMatrix_(squared_kernel_matrix_);
  }
  else {
    GrowMatrix_(kernel_matrix_);
  }

  // The kernel values.
  std::vector<double> kernel_values(dataset_->n_cols());

  // The optimal point information.
  int optimal_point_index = -1;
  double optimum_value = std::numeric_limit<double>::max();

  // Loop over candidates and decide to add the optimal.
  for (int i = 0; i < candidate_indices.size(); i++) {

    // Candidate index for which the kernel values have to be computed.
    int candidate_index = candidate_indices[i];
    ComputeKernelValues_(covariance_in, candidate_index, &kernel_values);

    // Fill in the matrix accordingly andn solve the optimization
    // problem.
    if (for_coeffs) {
      FillSquaredKernelMatrix_(squared_kernel_matrix_);
    }
    else {
      FillKernelMatrix_(kernel_matrix_);
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

  // The maximum number of points to choose in each iteration.
  const int max_num_points = 59;

  // Initialize the model.
  model_out->Init(dataset_, targets_);

  // Initialize the initial index sets to choose from (inactive
  // sets).
  std::vector<int> inactive_indices;
  std::vector<int> inactive_indices_for_error;

  do {

    // The candidate sets in the current iteration.
    std::vector<int> candidate_indices;
    std::vector<int> candidate_indices_for_error;

    // Choose a random subset from the inactive point set.
    ChooseRandomSubset_(
      inactive_indices, max_num_points, &candidate_indices);
    ChooseRandomSubset_(
      inactive_indices_for_error,
      max_num_points,
      &candidate_indices_for_error);

    // Choose a random optimal point for both sets.
    model_out->AddOptimalPoint(
      covariance_in, noise_level_in, candidate_indices, false);
    model_out->AddOptimalPoint(
      covariance_in, noise_level_in, candidate_indices_for_error, true);

  }
  while (Done_());
}
};
};

#endif
