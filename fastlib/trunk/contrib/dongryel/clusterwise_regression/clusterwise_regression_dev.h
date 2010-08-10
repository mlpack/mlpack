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
double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint, int cluster_number) const {

  double prediction = 0;

  return prediction;
}

double ClusterwiseRegressionResult::Predict(
  const Vector &datapoint) const {

  double mixed_prediction = 0;
  for (int i = 0; i < membership_probabilities_.length(); i++) {
    mixed_prediction += membership_probabilities_[i] *
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
  for (int j = 0; j < dataset_in.n_cols(); j++) {
    for (int i = 0; i < dataset_in.n_rows(); i++) {
      ranges[i] |= dataset_in.get(i, j);
    }
  }
  double diameter = 0;
  for (int i = 0; i < ranges.size(); i++) {
    diameter += math::Sqr(ranges[i].width());
  }
  return diameter;
}

void ClusterwiseRegressionResult::Init(
  const Matrix *dataset_in, int num_clusters_in) {

  num_clusters_ = num_clusters_in;
  membership_probabilities_.Init(num_clusters_);
  coefficients_.Init(dataset_in->n_rows() + 1, num_clusters_);
  bandwidths_.resize(num_clusters_);

  // Initialize the parameters. Randomly assign a point to a
  // cluster.
  int random_cluster_number = math::RandInt(num_clusters_in);
  membership_probabilities_.SetZero();
  membership_probabilities_[random_cluster_number] = 1.0;

  // Initialize the coefficients.
  coefficients_.SetZero();

  // Initialize the bandwidths.
  double diameter = Diameter_(*dataset_in);
  for (int i = 0; i < num_clusters_in; i++) {
    bandwidth_[i] = math::Random(0.1 * diameter, 0.5 * diameter);
  }
}
};

namespace ml {
void ClusterwiseRegression::EStep_(ClusterwiseRegressionResult &result_out) {

  // Compute the MLE of the posterior probability for each point for
  // each cluster.
  for (int j = 0; j < membership_probabilities_.n_cols(); j++) {
    for (int i = 0; i < membership_probabilities_.n_rows(); i++) {

    }
  }
}

void ClusterwiseRegression::MStep_(ClusterwiseRegressionResult &result_out) {

}

ClusterwiseRegression::ClusterwiseRegression() {
  dataset_ = NULL;
}

void ClusterwiseRegression::Init(const Matrix &dataset_in) {
  dataset_ = &dataset_in;
}

void ClusterwiseRegression::Compute(
  int num_clusters_in,
  ClusterwiseRegressionResult *result_out) {

  result_out->Init(dataset_, num_clusters_in);

  // Run the EM algorithm to estimate the (probabilistic) membership
  // and the parameters.
  do {

    // The E-step.
    EStep_(*result_out);

    // The M-step.
    MStep_(*result_out);

  }
  while (Converged_() == false);
}
};

#endif
