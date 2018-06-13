/**
 * @file lmnn.hpp
 * @author Manish Kumar
 *
 * Declaration of Large Margin Nearest Neighbor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_LMNN_HPP
#define MLPACK_METHODS_LMNN_LMNN_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>

#include "lmnn_function.hpp"

namespace mlpack {
namespace lmnn /** Large Margin Nearest Neighbor. */ {

/**
 * An implementation of Large Margin nearest neighbor metric learning technique.
 * The method seeks to improve clustering & classification algorithms on
 * a dataset by transforming the dataset representation in a more convenient
 * form for them. It introduces the concept of target neighbors and impostors,
 * focusing on the idea that the distance between impostors and the perimeters
 * established by target neighbors should be large and that between target
 * neighbors and data point should be small. It requires the knowledge of
 * target neighbors beforehand. Moreover, target neighbors once initialized
 * remain same.
 *
 * @tparam SDPType The type of SDP to use for computation. ex. SDP<arma::mat>
 */
template<typename MetricType = metric::SquaredEuclideanDistance,
         typename OptimizerType = optimization::AMSGrad>
class LMNN
{
 public:
  /**
   * Initialize the LMNN object, passing a dataset (distance metric
   * is learned using this dataset) and labels. Initialization will copy
   * both dataset and labels matrices to internal copies.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   */
  LMNN(const arma::mat& dataset,
       const arma::Row<size_t>& labels,
       const size_t k,
       const MetricType metric = MetricType());


  /**
   * Perform Large Margin Nearest Neighbors metric learning. The output
   * distance matrix is written into the passed reference. If the
   * LearnDistance() is called with an outputMatrix with correct dimensions,
   * then that matrix will be used as the starting point for optimization.
   *
   * @param outputMatrix Covariance matrix of Mahalanobis distance.
   */
  void LearnDistance(arma::mat& outputMatrix);


  //! Get the dataset reference.
  const arma::mat& Dataset() const { return dataset; }

  //! Get the labels reference.
  const arma::Row<size_t>& Labels() const { return labels; }

  //! Access the regularization value.
  const double& Regularization() const { return objFunction.Regularization(); }
  //! Modify the regularization value.
  double& Regularization() { return objFunction.Regularization(); }

  //! Access the range value.
  const size_t& Range() const { return objFunction.Range(); }
  //! Modify the range value.
  size_t& Range() { return objFunction.Range(); }

  //! Access the value of k.
  const size_t& K() const { return k; }

  //! Get the optimizer.
  const OptimizerType& Optimizer() const { return optimizer; }
  OptimizerType& Optimizer() { return optimizer; }

 private:
  //! Dataset reference.
  const arma::mat& dataset;

  //! Labels reference.
  const arma::Row<size_t>& labels;

  const size_t k;

  //! Metric to be used.
  MetricType metric;

  //! The optimizer to use.
  OptimizerType optimizer;

  // LMNN objective function.
  LMNNFunction<MetricType> objFunction;
}; // class LMNN

} // namespace lmnn
} // namespace mlpack

// Include the implementation.
#include "lmnn_impl.hpp"

#endif
