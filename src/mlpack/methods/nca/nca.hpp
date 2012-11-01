/**
 * @file nca.hpp
 * @author Ryan Curtin
 *
 * Declaration of NCA class (Neighborhood Components Analysis).
 */
#ifndef __MLPACK_METHODS_NCA_NCA_HPP
#define __MLPACK_METHODS_NCA_NCA_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace nca /** Neighborhood Components Analysis. */ {

/**
 * An implementation of Neighborhood Components Analysis, both a linear
 * dimensionality reduction technique and a distance learning technique.  The
 * method seeks to improve k-nearest-neighbor classification on a dataset by
 * scaling the dimensions.  The method is nonparametric, and does not require a
 * value of k.  It works by using stochastic ("soft") neighbor assignments and
 * using optimization techniques over the gradient of the accuracy of the
 * neighbor assignments.
 *
 * For more details, see the following published paper:
 *
 * @code
 * @inproceedings{Goldberger2004,
 *   author = {Goldberger, Jacob and Roweis, Sam and Hinton, Geoff and
 *       Salakhutdinov, Ruslan},
 *   booktitle = {Advances in Neural Information Processing Systems 17},
 *   pages = {513--520},
 *   publisher = {MIT Press},
 *   title = {{Neighbourhood Components Analysis}},
 *   year = {2004}
 * }
 * @endcode
 */
template<typename MetricType>
class NCA
{
 public:
  /**
   * Construct the Neighborhood Components Analysis object.  This simply stores
   * the reference to the dataset and labels as well as the parameters for
   * optimization before the actual optimization is performed.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param stepSize Step size for stochastic gradient descent.
   * @param maxIterations Maximum iterations for stochastic gradient descent.
   * @param tolerance Tolerance for termination of stochastic gradient descent.
   * @param shuffle Whether or not to shuffle the dataset during SGD.
   * @param metric Instantiated metric to use.
   */
  NCA(const arma::mat& dataset,
      const arma::uvec& labels,
      const double stepSize = 0.01,
      const size_t maxIterations = 500000,
      const double tolerance = 1e-10,
      const bool shuffle = true,
      MetricType metric = MetricType());

  /**
   * Perform Neighborhood Components Analysis.  The output distance learning
   * matrix is written into the passed reference.  If LearnDistance() is called
   * with an outputMatrix which has the correct size (dataset.n_rows x
   * dataset.n_rows), that matrix will be used as the starting point for
   * optimization.
   *
   * @param output_matrix Covariance matrix of Mahalanobis distance.
   */
  void LearnDistance(arma::mat& outputMatrix);

  //! Get the dataset reference.
  const arma::mat& Dataset() const { return dataset; }
  //! Get the labels reference.
  const arma::uvec& Labels() const { return labels; }

  //! Get the step size for stochastic gradient descent.
  double StepSize() const { return stepSize; }
  //! Modify the step size for stochastic gradient descent.
  double& StepSize() { return stepSize; }

  //! Get the maximum number of iterations for stochastic gradient descent.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations for stochastic gradient descent.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for the termination of stochastic gradient descent.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for the termination of stochastic gradient descent.
  double& Tolerance() { return tolerance; }

 private:
  //! Dataset reference.
  const arma::mat& dataset;
  //! Labels reference.
  const arma::uvec& labels;

  //! Metric to be used.
  MetricType metric;

  //! Step size for stochastic gradient descent.
  double stepSize;
  //! Maximum iterations for stochastic gradient descent.
  size_t maxIterations;
  //! Tolerance for termination of stochastic gradient descent.
  double tolerance;
  //! Whether or not to shuffle the dataset for SGD.
  bool shuffle;
};

}; // namespace nca
}; // namespace mlpack

// Include the implementation.
#include "nca_impl.hpp"

#endif
