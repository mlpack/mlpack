/** @file clusterwise_regression_dev.h
 *
 *  @brief An implementation of clusterwise regression via the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEV_H
#define ML_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEV_H

namespace ml {
ClusterwiseRegressionResult::ClusterwiseRegressionResult() {
  num_clusters_ = 0;
}

void ClusterwiseRegressionResult::Init(
  const Matrix *dataset_in, int num_clusters_in) {

  num_clusters_ = num_clusters_in;
  membership_probabilities_.Init(dataset_in->n_cols(), num_clusters_);
  coefficients_.Init(dataset_in->n_rows() + 1, num_clusters_);
  bandwidths_.resize(num_clusters_);

  // Initialize the parameters. Randomly assign a point to a
  // cluster.
  for (int i = 0; i < dataset_in->n_cols(); i++) {
    int random_cluster_number = math::RandInt(num_clusters_in);
    for (int j = 0; j < num_clusters_in; j++) {
      if (j == random_cluster_number) {
        membership_probabilities_.set(i, j, 1.0);
      }
      else {
        membership_probabilities_.set(i, j, 0.0);
      }
    }
  }
  coefficients_.SetZero();
  for (int i = 0; i < num_clusters_in; i++) {
    bandwidth_[i] = ;
  }
}
};

namespace ml {
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
}
};

#endif
