/** @file clusterwise_regression_dev.h
 *
 *  @brief An implementation of clusterwise regression via the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEV_H
#define MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEV_H

#include <vector>
#include "fastlib/fastlib.h"
#include "clusterwise_regression_defs.h"

namespace fl {
namespace ml {

double ClusterwiseRegressionResult::mixture_weight(int cluster_number) const {
  return mixture_weights_[cluster_number];
}

void ClusterwiseRegressionResult::set_mixture_weight(
  int cluster_number, double new_weight) {
  mixture_weights_[cluster_number] = new_weight;
}

int ClusterwiseRegressionResult::num_clusters() const {
  return kernels_.size();
}

void ClusterwiseRegressionResult::get_parameters(
  Vector *parameters_out) const {

  // Copy the linear regression coefficients for all clusters.
  int indexing_position = 0;
  for (int j = 0; j < coefficients_.n_cols(); j++) {
    for (int i = 0; i < coefficients_.n_rows(); i++, indexing_position++) {
      (*parameters_out)[indexing_position] = coefficients_.get(i, j);
    }
  }

  // Copy the mixture weights.
  for (int k = 0; k < mixture_weights_.length(); k++, indexing_position++) {
    (*parameters_out)[indexing_position] = mixture_weights_[k];
  }

  // Copy the squared bandwidth values.
  for (unsigned int k = 0; k < kernels_.size(); k++, indexing_position++) {
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

  double prediction = la::Dot(
                        datapoint.length(), datapoint.ptr(),
                        coefficients_.GetColumnPtr(cluster_number)) +
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
  for (int cluster_number = 0; cluster_number < mixture_weights_.length();
       cluster_number++) {
    mixed_prediction += mixture_weights_[cluster_number] *
                        Predict(datapoint, cluster_number);
  }
  return mixed_prediction;
}

double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint, double target, double *squared_error) const {

  double prediction = Predict(datapoint);
  *squared_error = math::Sqr(target - prediction);
  return prediction;
}

ClusterwiseRegressionResult::ClusterwiseRegressionResult() {
  num_clusters_ = 0;
}

double ClusterwiseRegressionResult::Diameter_(const Matrix &dataset_in) const {
  std::vector< DRange > ranges(dataset_in.n_rows());
  for (unsigned int i = 0; i < ranges.size(); i++) {
    ranges[i].InitEmptySet();
  }
  for (int j = 0; j < dataset_in.n_cols(); j++) {
    for (int i = 0; i < dataset_in.n_rows(); i++) {
      ranges[i] |= dataset_in.get(i, j);
    }
  }
  double diameter = 0;
  for (unsigned int i = 0; i < ranges.size(); i++) {
    diameter += math::Sqr(ranges[i].width());
  }
  diameter = sqrt(diameter);
  return diameter;
}

void ClusterwiseRegressionResult::Init(
  const Matrix *dataset_in, int num_clusters_in) {

  num_clusters_ = num_clusters_in;
  mixture_weights_.Init(num_clusters_);
  coefficients_.Init(dataset_in->n_rows() + 1, num_clusters_);
  kernels_.resize(num_clusters_);

  // Initialize the parameters.
  double mixture_sum = 0;
  for (int i = 0; i < mixture_weights_.length(); i++) {
    mixture_weights_[i] = math::Random(0.1, 0.5);
    mixture_sum += mixture_weights_[i];
  }
  for (int i = 0; i < mixture_weights_.length(); i++) {
    mixture_weights_[i] /= mixture_sum;
  }

  // Initialize the coefficients.
  for (int j = 0; j < coefficients_.n_cols(); j++) {
    for (int i = 0; i < coefficients_.n_rows(); i++) {
      coefficients_.set(i, j, math::Random(-0.5, 0.5));
    }
  }

  // Initialize the bandwidths.
  double diameter = Diameter_(*dataset_in);
  for (int i = 0; i < num_clusters_in; i++) {
    kernels_[i].Init(math::Random(0.3 * diameter, 0.75 * diameter));
  }
}
};

namespace ml {

void ClusterwiseRegression::Solve_(
  const Matrix &membership_probabilities,
  int cluster_number, Vector *solution_out) {

  // The right hand side weighted by the probabilities.
  Matrix weighted_right_hand_side;
  weighted_right_hand_side.Init(targets_->length(), 1);
  for (int i = 0; i < targets_->length(); i++) {
    weighted_right_hand_side.set(
      i, 0, (*targets_)[i] * sqrt(
        membership_probabilities.get(i, cluster_number)));
  }

  // The left hand side weighted by the probabilities.
  Matrix weighted_left_hand_side;
  weighted_left_hand_side.Init(dataset_->n_cols(), dataset_->n_rows() + 1);
  for (int i = 0; i < dataset_->n_cols(); i++) {
    Vector point;
    dataset_->MakeColumnVector(i, &point);
    for (int j = 0; j < dataset_->n_rows(); j++) {
      weighted_left_hand_side.set(
        i, j, point[j] *
        sqrt(membership_probabilities.get(i, cluster_number)));
    }
    weighted_left_hand_side.set(
      i, dataset_->n_rows(),
      sqrt(membership_probabilities.get(i, cluster_number)));
  }

  // Take the QR and transform the right hand side.
  Matrix q_factor, r_factor;
  la::QRInit(weighted_left_hand_side, &q_factor, &r_factor);
  Matrix q_trans_right_hand_side_mat;
  Vector q_trans_right_hand_side;
  la::MulTransAInit(
    q_factor, weighted_right_hand_side, &q_trans_right_hand_side_mat);
  q_trans_right_hand_side_mat.MakeColumnVector(0, &q_trans_right_hand_side);

  // SVD the R factor and solve it.
  Vector singular_values;
  Matrix left_singular_vectors, right_singular_vectors_transposed;
  la::SVDInit(
    r_factor, &singular_values, &left_singular_vectors,
    &right_singular_vectors_transposed);

  solution_out->SetZero();
  for (int j = 0; j < singular_values.length(); j++) {
    if (singular_values[j] > 1e-6) {
      Vector left_singular_vector;
      left_singular_vectors.MakeColumnVector(j, &left_singular_vector);
      double scaling_factor =
        la::Dot(
          left_singular_vector, q_trans_right_hand_side) / singular_values[j];
      for (int k = 0; k < right_singular_vectors_transposed.n_cols(); k++) {
        (*solution_out)[k] += scaling_factor *
                              right_singular_vectors_transposed.get(j, k);
      }
    }
  }
}

void ClusterwiseRegression::UpdateMixture_(
  const Matrix &membership_probabilities,
  int cluster_number, ClusterwiseRegressionResult &result_out) {

  double sum_probabilities = 0;
  double weighted_sum_probabilities = 0;

  // Loop through each data point.
  for (int i = 0; i < dataset_->n_cols(); i++) {

    // Do a prediction for the current point on the current cluster
    // model.
    Vector point;
    dataset_->MakeColumnVector(i, &point);
    double squared_error;
    result_out.Predict(
      point, (*targets_)[i],
      cluster_number, &squared_error);
    weighted_sum_probabilities += membership_probabilities.get(
                                    i, cluster_number) * squared_error;
    sum_probabilities += membership_probabilities.get(i, cluster_number);
  }
  result_out.kernel(cluster_number).Init(
    sqrt(weighted_sum_probabilities / sum_probabilities));
}

void ClusterwiseRegression::EStep_(
  const ClusterwiseRegressionResult &result_out,
  Matrix &membership_probabilities) {

  // Compute the MLE of the posterior probability for each point for
  // each cluster.
  Vector probabilities_per_point;
  probabilities_per_point.Init(result_out.num_clusters());
  for (int k = 0; k < dataset_->n_cols(); k++) {

    // Get the point.
    Vector point;
    dataset_->MakeColumnVector(k, &point);
    double normalization = 0;

    // Compute the probability for each cluster.
    for (int j = 0; j < result_out.num_clusters(); j++) {
      double squared_error;
      result_out.Predict(point, (*targets_)[k], j, &squared_error);
      probabilities_per_point[j] =
        result_out.mixture_weight(j) *
        result_out.kernel(j).EvalUnnormOnSq(squared_error) /
        result_out.kernel(j).CalcNormConstant(point.length());
      normalization += probabilities_per_point[j];
    } // end of looping through each cluster.

    for (int j = 0; j < result_out.num_clusters(); j++) {
      membership_probabilities.set(
        k, j, probabilities_per_point[j] / normalization);
    } // end of looping through each cluster.

  } // end of looping through each point.
}

void ClusterwiseRegression::MStep_(
  const Matrix &membership_probabilities,
  ClusterwiseRegressionResult &result_out) {

  // Update the mixture weights.
  for (int i = 0; i < result_out.num_clusters(); i++) {
    double sum = 0;
    for (int j = 0; j < dataset_->n_cols(); j++) {
      sum += membership_probabilities.get(j, i);
    }
    result_out.set_mixture_weight(i, sum / ((double) dataset_->n_cols()));
  }

  // Update the linear model for each cluster.
  Matrix &coefficients = result_out.coefficients();
  for (int j = 0; j < result_out.num_clusters(); j++) {
    Vector coefficients_per_cluster;
    coefficients.MakeColumnVector(j, &coefficients_per_cluster);
    Solve_(membership_probabilities, j, &coefficients_per_cluster);
  }

  // Update the variances of each mixture.
  for (int j = 0; j < result_out.num_clusters(); j++) {
    UpdateMixture_(membership_probabilities, j, result_out);
  }
}

ClusterwiseRegression::ClusterwiseRegression() {
  dataset_ = NULL;
  targets_ = NULL;
}

void ClusterwiseRegression::Init(
  const Matrix &dataset_in, const Vector &targets_in) {
  dataset_ = &dataset_in;
  targets_ = &targets_in;
}

bool ClusterwiseRegression::Converged_(
  const Vector &parameters_before_update,
  const Vector &parameters_after_update) const {

  double norm_before_update = sqrt(
                                la::Dot(
                                  parameters_before_update, parameters_before_update));
  double norm_after_update = sqrt(
                               la::Dot(
                                 parameters_after_update, parameters_after_update));

  return fabs(norm_after_update - norm_before_update) <= 1e-6;
}

void ClusterwiseRegression::Compute(
  int num_clusters_in,
  int num_iterations_in,
  ClusterwiseRegressionResult *result_out) {

  result_out->Init(dataset_, num_clusters_in);

  // Run the EM algorithm to estimate the (probabilistic) membership
  // and the parameters.
  int iteration_number = 0;
  Vector parameters_before_update;
  Vector parameters_after_update;
  parameters_before_update.Init(num_clusters_in *(dataset_->n_rows() + 3));
  parameters_after_update.Init(num_clusters_in *(dataset_->n_rows() + 3));

  // Initialize the membership probabilities.
  Matrix membership_probabilities;
  membership_probabilities.Init(dataset_->n_cols(), num_clusters_in);

  do {

    // Get the parameters before the E and the M steps.
    result_out->get_parameters(&parameters_before_update);

    // The E-step.
    EStep_(*result_out, membership_probabilities);

    // The M-step.
    MStep_(membership_probabilities, *result_out);

    // Get the parameters after the E and the M steps.
    result_out->get_parameters(&parameters_after_update);

    // Increment the iteration number.
    iteration_number++;
  }
  while (Converged_(
           parameters_before_update, parameters_after_update) == false &&
         iteration_number < num_iterations_in);
}
};
};

#endif
