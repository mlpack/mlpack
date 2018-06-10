/**
 * @file lmnn_function.hpp
 * @author Manish Kumar
 *
 * Declaration of the LMNNFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_LMNN_FUNCTION_HPP
#define MLPACK_METHODS_LMNN_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace lmnn {


template<typename MetricType = metric::SquaredEuclideanDistance>
class LMNNFunction
{
 public:
  /**
   * Constructor for LMNNFunction class.
   *
   * @param data Dataset for which metric is calculated.
   * @param rank Rank used for matrix factorization.
   * @param lambda Regularization parameter used for optimization.
   */
  LMNNFunction(const arma::mat& dataset,
               const arma::Row<size_t>& labels,
               size_t k,
               double regularization,
               MetricType metric = MetricType());

  /**
   * Shuffle the points in the dataset. This may be used by optimizers.
   */
  void Shuffle();

  /**
   * Evaluate the LMNN function for the given covariance matrix.  This is the
   * non-separable implementation, where the objective function is not
   * decomposed into the sum of several objective functions.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   */
  double Evaluate(const arma::mat& covariance);

  /**
   * Evaluate the LMNN objective function for the given covariance matrix on
   * the given batch size from a given inital point of the dataset.
   * This is the separable implementation, where the objective 
   * function is decomposed into the sum of many objective
   * functions, and here, only one of those constituent objective functions is
   * returned.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param begin Index of the initial point to use for objective function.
   * @param batchSize Number of points to use for objective function.
   */
  double Evaluate(const arma::mat& covariance,
                  const size_t begin,
                  const size_t batchSize = 1);

  /**
   * Evaluate the gradient of the LMNN function for the given covariance
   * matrix.  This is the non-separable implementation, where the objective
   * function is not decomposed into the sum of several objective functions.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  template<typename GradType>
  void Gradient(const arma::mat& covariance, GradType& gradient);

  /**
   * Evaluate the gradient of the LMNN function for the given covariance
   * matrix on the given batch size, from a given initial point of the dataset.
   * This is the separable implementation, where the objective function is
   * decomposed into the sum of many objective functions, and here,
   * only one of those constituent objective functions is returned.
   * The type of the gradient parameter is a template
   * argument to allow the computation of a sparse gradient.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param begin Index of the initial point to use for objective function.
   * @param batchSize Number of points to use for objective function.
   * @param gradient Matrix to store the calculated gradient in.
   */
  template<typename GradType>
  void Gradient(const arma::mat& covariance,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize = 1);

  /**
   * Evaluate the LMNN objective function together with gradient for the given
   * covariance matrix.  This is the non-separable implementation, where the
   * objective function is not decomposed into the sum of several objective
   * functions.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& coordinates,
                            GradType& gradient);

  /**
   * Evaluate the LMNN objective function together with gradient for the given
   * covariance matrix on the given batch size, from a given initial point of
   * the dataset. This is the separable implementation, where the objective
   * function is decomposed into the sum of many objective functions, and
   * here, only one of those constituent objective functions is returned.
   * The type of the gradient parameter is a template
   * argument to allow the computation of a sparse gradient.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param begin Index of the initial point to use for objective function.
   * @param batchSize Number of points to use for objective function.
   * @param gradient Matrix to store the calculated gradient in.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& covariance,
                            const size_t begin,
                            GradType& gradient,
                            const size_t batchSize = 1);

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  /**
   * Get the number of functions the objective function can be decomposed into.
   * This is just the number of points in the dataset.
   */
  size_t NumFunctions() const { return dataset.n_cols; }

  //! Return the dataset passed into the constructor.
  const arma::mat& Dataset() const { return dataset; }

  //! Access the regularization value.
  const double& Regularization() const { return regularization; }
  //! Modify the regularization value.
  double& Regularization() { return regularization; }

  //! Access the value of k.
  const size_t& K() const { return k; }
  //! Modify the value of k.
  size_t& K() { return k; }

 private:
  //! data.  This will be an alias until Shuffle() is called.
  arma::mat dataset;
  //! labels.  This will be an alias until Shuffle() is called.
  arma::Row<size_t> labels;
  //! Initial parameter point.
  arma::mat initialPoint;
  //! Store transformed dataset.
  arma::mat transformedDataset;
  //! Store target neighbors of data points.
  arma::Mat<size_t> targetNeighbors;
  //! Initial impostors.
  arma::Mat<size_t> impostors;
  //! Number of target neighbors.
  size_t k;
  //! The instantiated metric.
  MetricType metric;
  //! Regularization value.
  double regularization;
  //! Holds pre-calculated cij.
  std::vector<arma::mat> p_cij;
  //! False if nothing has ever been precalculated.
  bool precalculated;
  /**
  * Precalculate the gradient part due to target neighbors and stores
  * the contribution of each data point as a matrix of matrices p_cij
  * for each batch. In case of L-BFGS batch size will be equal to total
  * number of data points.
  * @param batchSize Number of points to use for each p_cij internal matrix.
  */
  void Precalculate(const size_t batchSize);
};

} // namespace lmnn
} // namespace mlpack

#include "lmnn_function_impl.hpp"

#endif
