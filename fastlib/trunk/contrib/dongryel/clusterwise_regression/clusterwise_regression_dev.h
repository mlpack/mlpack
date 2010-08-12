/** @file clusterwise_regression_dev.h
 *
 *  @brief An implementation of clusterwise regression via the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEV_H
#define ML_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEV_H

#include <vector>
#include "fastlib/fastlib.h"

namespace ml {

void ClusterwiseRegressionResult::get_parameters(
  Vector *parameters_out) const {

  int indexing_position = 0;
  for(int j = 0; j < coefficients_.n_cols(); j++) {
    for(int i = 0; i < coefficients_.n_rows(); i++) {
      (*parameters_out)[indexing_position] = coefficients_.get(i, j);
      indexing_position++;
    }
  }
  for(int k = 0; k < mixture_weights_.length(); k++, indexing_position++) {
    (*parameters_out)[indexing_position] = mixture_weights_[k];
  }
  for(int k = 0; k < kernels_.size(); k++, indexing_position++) {
    (*parameters_out)[indexing_position] = kernels_[k].bandwidth_sq();
  }
}

const GaussianKernel &ClusterwiseRegressionResult::kernel(
  int cluster_number) const {

  return kernels_[cluster_number];
}

GaussianKernel &ClusterwiseRegressionResult::kernel(int cluster_number) {
  return kernels_[cluster_number];
}

const Matrix &ClusterwiseRegressionResult::coefficients() const {
  return coefficients_;
}

Matrix &ClusterwiseRegressionResult::coefficients() {
  return coefficients_;
}

double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint, int cluster_number) const {

  Vector coeff_alias;
  coeff_alias.MakeAlias(
    coefficients_.GetColumnPtr(cluster_number), datapoint.length());
  double prediction = la::Dot(datapoint, coeff_alias) +
                      coefficients_.get(datapoint.length(), cluster_number);
  return prediction;
}

double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint, double target,
  int cluster_number, double *squared_error) const {

  double prediction = Predict(datapoint, cluster_number);
  *squared_error = math::Sqr(prediction - target);
  return prediction;
}

double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint) const {

  double mixed_prediction = 0;
  for(int i = 0; i < mixture_weights_.length(); i++) {
    mixed_prediction += mixture_weights_[i] *
                        Predict(datapoint, cluster_number);
  }
  return mixed_prediction;
}

double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint, double target, double *squared_error) const {

  double prediction = Predict(datapoint);
  *squared_error = fl::math::Sqr(target - prediction);
  return prediction;
}

ClusterwiseRegressionResult::ClusterwiseRegressionResult() {
  num_clusters_ = 0;
}

double ClusterwiseRegressionResult::Diameter_(const Matrix &dataset_in) const {
  std::vector< DRange > ranges;
  ranges.resize(dataset_in.n_rows());
  for(int j = 0; j < dataset_in.n_cols(); j++) {
    for(int i = 0; i < dataset_in.n_rows(); i++) {
      ranges[i] |= dataset_in.get(i, j);
    }
  }
  double diameter = 0;
  for(int i = 0; i < ranges.size(); i++) {
    diameter += math::Sqr(ranges[i].width());
  }
  diameter = sqrt(diameter);
  return diameter;
}

void ClusterwiseRegressionResult::Init(
  const Matrix *dataset_in, int num_clusters_in) {

  num_clusters_ = num_clusters_in;
  mixture_weights_.Init(num_clusters_);
  membership_probabilities_.Init(dataset_in->n_cols(), num_clusters_in);
  coefficients_.Init(dataset_in->n_rows() + 1, num_clusters_);
  kernels_.resize(num_clusters_);

  // Initialize the parameters.
  int random_cluster_number = math::RandInt(num_clusters_in);
  mixture_weights_.SetZero();
  mixture_weights_[random_cluster_number] = 1.0;

  // Initialize the coefficients.
  membership_probabilities_.SetZero();
  coefficients_.SetZero();

  // Initialize the bandwidths.
  double diameter = Diameter_(*dataset_in);
  for(int i = 0; i < num_clusters_in; i++) {
    kernels_[i].Init(math::Random(0.1 * diameter, 0.5 * diameter));
  }
}
};

namespace ml {

void ClusterwiseRegression::Solve_(int cluster_number, Vector *solution_out) {

  // The right hand side weighted by the probabilities.
  Vector right_hand_side;
  right_hand_side.Init(targets_.length());
  for(int i = 0; i < targets_.length(); i++) {
    weighted_targets_[i] = targets_[i] * sqrt(
                             membership_probabilities_.get(i, cluster_number));
  }

  // The left hand side weighted by the probabilities.
  Matrix weighted_left_hand_side;
  weighted_left_hand_side.Init(dataset_->n_cols(), dataset_->n_rows() + 1);
  for(int i = 0; i < dataset_->n_cols(); i++) {
    Vector point;
    dataset_->MakeColumnVector(i, &point);
    for(int j = 0; j < dataset_->n_rows(); j++) {
      weighted_left_hand_side.set(
        i, j, point[j] *
        sqrt(membership_probabilities_.get(i, cluster_number)));
    }
    weighted_left_hand_side.set(
      i, dataset_->n_rows(),
      sqrt(membership_probabilities_.get(i, cluster_number)));
  }

  // Take the QR and transform the right hand side.
  Matrix q_factor, r_factor;
  la::QRInit(weighted_left_hand_side, &q_factor, &r_factor);
  Vector q_trans_right_hand_side;
  la::MulTransAInit(q_factor, right_hand_side, &q_trans_right_hand_side);

  // SVD the R factor and solve it.
  Matrix singular_values, left_singular_vectors,
         right_singular_vectors_transposed;
  la::SVDInit(
    r_factor, &singular_values, &left_singular_vectors,
    &right_singular_vectors_transposed);

  solution_out->SetZero();
  for(int j = 0; j < singular_values.length(); j++) {
    if(singular_values[j] > 1e-6) {
      Vector left_singular_vector;
      left_singular_vectors.MakeColumnVector(j, &left_singular_vector);
      double scaling_factor =
        la::Dot(
          left_singular_vector, q_trans_right_hand_side) / singular_values[j];
      for(int k = 0; k < right_singular_vectors_transposed.length(); k++) {
        (*solution_out)[k] += scaling_factor *
                              right_singular_vectors_transposed.get(j, k);
      }
    }
  }
}

void ClusterwiseRegression::UpdateMixture_(
  int cluster_number, ClusterwiseRegressionResult &result_out) {

  double sum_probabilities = 0;
  double weighted_sum_probabilities = 0;

  // Loop through each data point.
  for(int i = 0; i < dataset_->n_cols(); i++) {

    // Do a prediction for the current point on the current cluster
    // model.
    Vector point;
    dataset_->MakeColumnVector(i, &point);
    double squared_error;
    double prediction = result_out.Predict(
                          point, targets_[i], cluster_number, &squared_error);
    weighted_sum_probabilities += membership_probabilities_.get(
                                    i, cluster_number) * squared_error;
    sum_probabilities += membership_probabilities_.get(i, cluster_number);
  }
  result_out.kernel(cluster_number).Init(
    sqrt(weighted_sum_probabilities / sum_probabilities));
}

void ClusterwiseRegression::EStep_(ClusterwiseRegressionResult &result_out) {

  // Compute the MLE of the posterior probability for each point for
  // each cluster.
  Vector probabilities_per_point;
  probabilities_per_point.Init(kernels_.size());
  for(int k = 0; k < dataset_->n_cols(); k++) {

    // Get the point.
    Vector point;
    dataset_->MakeColumnVector(k, &point);
    doiuble normalization = 0;

    // Compute the probability for each cluster.
    for(int j = 0; j < kernels_.size(); j++) {
      double squared_error;
      double prediction = Predict(point, targets_[k], j, &squared_error);
      probabilities_per_point[j] = mixture_weights_[j] *
                                   kernels_[j].EvalUnnormOnSq(squared_error) /
                                   kernels_[j].CalcNormConstant(point.length());
      normalization += probabilities_per_point[j];
    } // end of looping through each cluster.

    for(int j = 0; j < kernels_.size(); j++) {
      membership_probabilities_.set(
        k, j, probabilities_per_point[j] / normalization);
    } // end of looping through each cluster.

  } // end of looping through each point.
}

void ClusterwiseRegression::MStep_(ClusterwiseRegressionResult &result_out) {

  // Update the mixture weights.
  for(int i = 0; i < kernels_.size(); i++) {
    double sum = 0;
    for(int j = 0; j < dataset_->n_cols(); j++) {
      sum += membereship_probabilities_.get(j, i);
    }
    mixture_weights_[i] = sum / ((double) dataset_->n_cols());
  }

  // Update the linear model for each cluster.
  Matrix &coefficients = result_out.coefficients();
  for(int j = 0; j < kernels_.size(); j++) {
    Vector coefficients_per_cluster;
    coefficients.MakeColumnVector(j, &coefficients_per_cluster);
    Solve_(j, &coefficients_per_cluster);
  }

  // Update the variances of each mixture.
  for(int j = 0; j < kernels_.size(); j++) {
    UpdateMixture_(j, result_out);
  }
}

ClusterwiseRegression::ClusterwiseRegression() {
  dataset_ = NULL;
}

void ClusterwiseRegression::Init(const Matrix &dataset_in) {
  dataset_ = &dataset_in;
}

bool ClusterwiseRegression::Converged_(
  const Vector &parameters_before_update,
  const Vector &parameters_after_update) const {

  double norm_before_update = sqrt(la::Dot(parameters_before_update));
  double norm_after_update = sqrt(la::Dot(parameters_after_update));

  return fabs(norm_after_update - norm_before_update) <=
         0.01 * norm_before_update;
}

void ClusterwiseRegression::Compute(
  int num_clusters_in,
  ClusterwiseRegressionResult *result_out) {

  result_out->Init(dataset_, num_clusters_in);

  // Run the EM algorithm to estimate the (probabilistic) membership
  // and the parameters.
  int iteration_number = 0;
  Vector parameters_before_update;
  Vector parameters_after_update;
  parameters_before_update.Init(num_clusters_in *(dataset_->n_rows() + 3));
  parameters_after_update.Init(num_clusters_in *(dataset_->n_rows() + 3));

  do {

    // Get the parameters before the E and the M steps.
    result_out->get_parameters(&parameters_before_update);

    // The E-step.
    EStep_(*result_out);

    // The M-step.
    MStep_(*result_out);

    // Get the parameters after the E and the M steps.
    result_out->get_parameters(&parameters_after_update);

    // Increment the iteration number.
    iteration_number++;
  }
  while(Converged_(
          parameters_before_update, parameters_after_update) == false &&
        iteration_number < num_iterations);
}
};

#endif
