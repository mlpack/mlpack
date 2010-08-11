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
  for (int i = 0; i < mixture_weights_.length(); i++) {
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
  for (int j = 0; j < dataset_in.n_cols(); j++) {
    for (int i = 0; i < dataset_in.n_rows(); i++) {
      ranges[i] |= dataset_in.get(i, j);
    }
  }
  double diameter = 0;
  for (int i = 0; i < ranges.size(); i++) {
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
  for (int i = 0; i < num_clusters_in; i++) {
    kernels_[i].Init(math::Random(0.1 * diameter, 0.5 * diameter));
  }
}
};

namespace ml {
void ClusterwiseRegression::EStep_(ClusterwiseRegressionResult &result_out) {

  // Compute the MLE of the posterior probability for each point for
  // each cluster.
  Vector probabilities_per_point;
  probabilities_per_point.Init( kernels_.size() );
  for (int k = 0; k < dataset_->n_cols(); k++) {

    // Get the point.
    Vector point;
    dataset_->MakeColumnVector(k, &point);
    doiuble normalization = 0;

    // Compute the probability for each cluster.
    for (int j = 0; j < kernels_.size(); j++) {
      double squared_error;
      double prediction = Predict(point, targets_, j, &squared_error);
      probabilities_per_point[j] = kernels_[j].EvalUnnormOnSq(squared_error) /
	kernels_[j].CalcNormConstant( point.length() );
      normalization += probabilities_per_point[j];
    } // end of looping through each cluster.

    for(int j = 0; j < kernels_.size(); j++) {
      membership_probabilities_.set(
        k, j, probabilities_per_point[j] / normalization);
    } // end of looping through each cluster.
 
  } // end of looping through each point.
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
