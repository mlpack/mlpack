/** @file clusterwise_regression.h
 *
 *  @brief A prototype of clusterwise regression via the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_H
#define ML_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_H

#include "fastlib/fastlib.h"

namespace ml {
class ClusterwiseRegressionResult {
  private:

    /** @brief The number of clusters.
     */
    int num_clusters_;

    /** @brief The membership probabilities for each data point. The
     *         dimensionality is the number of data points by the
     *         number of clusters.
     */
    Matrix membership_probabilities_;

    /** @brief The trained linear model for each cluster. The
     *         dimensionality is $D + 1$ by the number of clusters.
     */
    Matrix coefficients_;

    /** @brief The bandwidths per each cluster.
     */
    std::vector<double> bandwidths_;

  public:

    ClusterwiseRegressionResult();
};

class ClusterwiseRegression {
  private:
    const Matrix *dataset_;

  public:

    ClusterwiseRegression();

    void Init(const Matrix &dataset_in);

    void Compute(
      int num_clusters_in,
      ClusterwiseRegressionResult *result_out);
};
};

#endif
