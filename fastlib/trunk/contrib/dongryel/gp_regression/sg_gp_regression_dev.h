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

void SparseGreedyGprModel::SetupMatrix_(
  const std::vector< std::vector<double> > &matrix_in,
  Matrix *matrix_out) const {

  matrix_out->Init(matrix_in.size(), matrix_in.size());
  for (int j = 0; j < matrix_in.size(); j++) {
    for (int i = 0; i < matrix_in.size(); i++) {
      matrix_out->set(i, j, matrix_in[i][j]);
    }
  }
}

void SparseGreedyGprModel::AddOptimalPoint(
  const std::vector<int> &candidate_indices) {

  // The kernel values against the previously existing points.
  std::vector<double> kernel_values(inverse_.size());

  // Loop over candidates and decide to add the optimal.
  for (int i = 0; i < candidate_indices.size(); i++) {

    // Candidate index.
    int candidate_index = candidate_indices[i];


  }
}

void SparseGreedyGprModel::AddOptimalPointForError(
  const std::vector<int> &candidate_indices) {

  // The kernel values against the previously existing points.
  std::vector<double> kernel_values(inverse_for_error_.size());

  // Loop over candidates and decide to add the optimal.
  for (int i = 0; i < candidate_indices.size(); i++) {

    // Candidate index.
    int candidate_index = candidate_indices[i];


  }
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

void SparseGreedyGpr::Compute(
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
    model_out->AddOptimalPoint(candidate_indices);
    model_out->AddOptimalPointForError(candidate_indices_for_error);

  }
  while (Done_());
}
};
};

#endif
