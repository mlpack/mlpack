/** @file clusterwise_regression.h
 *
 *  @brief A prototype of clusterwise regression via the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_H
#define MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_H

#include "fastlib/fastlib.h"

namespace fl {
namespace ml {
class ClusterwiseRegressionResult {
  private:

    /** @brief The number of clusters.
     */
    int num_clusters_;

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

    int num_clusters() const;

    void get_parameters(Vector *parameters_out) const;

    double mixture_weight(int cluster_number) const;

    void set_mixture_weight(int cluster_number, double new_weight);

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

    void Init(const Matrix *dataset_in, int num_clusters_in);
};

class ClusterwiseRegression {
  private:
    const Matrix *dataset_;

    const Vector *targets_;

  private:

    bool Converged_(
      const Vector &parameters_before_update,
      const Vector &parameters_after_update) const;

    void EStep_(
      const ClusterwiseRegressionResult &result_out,
      Matrix &membership_probabilities);

    void MStep_(
      const Matrix &membership_probabilities,
      ClusterwiseRegressionResult &result_out);

    void Solve_(
      const Matrix &membership_probabilities,
      int cluster_number, Vector *solution_out);

    void UpdateMixture_(
      const Matrix &membership_probabilities,
      int cluster_number, ClusterwiseRegressionResult &result_out);

  public:

    ClusterwiseRegression();

    void Init(const Matrix &dataset_in, const Vector &targets_in);

    void Compute(
      int num_clusters_in,
      int num_iterations_in,
      ClusterwiseRegressionResult *result_out);

    static int RunAlgorithm(boost::program_options::variables_map &vm);

    static int Main(const std::vector<std::string> &args);
};
};
};

#endif
