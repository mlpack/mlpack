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

    /** @brief The membership probabilities for each point for each
     *         cluster. These are MLE of the posterior probability
     *         that each observation comes from each cluster. The
     *         dimensionality is the number of points by the number of
     *         clusters.
     */
    Matrix membership_probabilities_;

    /** @brief The trained linear model for each cluster. The
     *         dimensionality is $D + 1$ by the number of clusters.
     */
    Matrix coefficients_;

    /** @brief The mixing probabilities for each cluster.
      */
    Vector mixture_weights_;

    /** @brief The Gaussian kernels per each cluster.
     */
    std::vector< GaussianKernel > kernels_;

  private:
    double Diameter_(const Matrix &dataset) const;

  public:

    void get_parameters(Vector *parameters_out) const;

    const GaussianKernel &kernel(int cluster_number) const;

    GaussianKernel &kernel(int cluster_number);

    const Matrix &coefficients() const;

    Matrix &coefficients();

    double Predict(const Vector &datapoint, int cluster_number) const;

    double Predict(
      const Vector &datapoint, double target,
      int cluster_number, double *squared_error) const;

    double Predict(const Vector &datapoint) const;

    double Predict(
      const Vector &datapoint, double target, double *squared_error) const;

    ClusterwiseRegressionResult();
};

class ClusterwiseRegression {
  private:
    const Matrix *dataset_;

    const Vector *targets_;

  private:

    bool Converged_(
      const Vector &parameters_before_update,
      const Vector &parameters_after_update) const;

    void EStep_(ClusterwiseRegressionResult &result_out);

    void MStep_(ClusterwiseRegressionResult &result_out);

    void Solve_(int cluster_number, Vector *solution_out);

    void UpdateMixture_(
      int cluster_number, ClusterwiseRegressionResult &result_out);

  public:

    ClusterwiseRegression();

    void Init(const Matrix &dataset_in, const Vector &targets_in);

    void Compute(
      int num_clusters_in,
      ClusterwiseRegressionResult *result_out);
};
};

#endif
