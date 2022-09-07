/**
 * @file methods/lmnn/lmnn.hpp
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

#include <mlpack/core.hpp>

#include "constraints.hpp"
#include "lmnn_function.hpp"

namespace mlpack {

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
 * For more details, see the following published paper:
 *
 * @code
 * @ARTICLE{weinberger09distance,
 *   author = {Weinberger, K.Q. and Saul, L.K.},
 *   title = {{Distance metric learning for large margin nearest neighbor
 *       classification}},
 *   journal = {The Journal of Machine Learning Research},
 *   year = {2009},
 *   volume = {10},
 *   pages = {207--244},
 *   publisher = {MIT Press}
 * }
 * @endcode
 *
 * @tparam MetricType The type of metric to use for computation.
 * @tparam OptimizerType Optimizer to use for developing distance.
 */
template<typename MetricType = SquaredEuclideanDistance,
         typename OptimizerType = ens::AMSGrad>
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
   * @param k Number of targets to consider.
   * @param metric Type of metric used for computation.
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
   * @tparam CallbackTypes Types of Callback functions.
   * @param outputMatrix Covariance matrix of Mahalanobis distance.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   */
  template<typename... CallbackTypes>
  void LearnDistance(arma::mat& outputMatrix, CallbackTypes&&... callbacks);


  //! Get the dataset reference.
  const arma::mat& Dataset() const { return dataset; }

  //! Get the labels reference.
  const arma::Row<size_t>& Labels() const { return labels; }

  //! Access the regularization value.
  const double& Regularization() const { return regularization; }
  //! Modify the regularization value.
  double& Regularization() { return regularization; }

  //! Access the range value.
  const size_t& Range() const { return range; }
  //! Modify the range value.
  size_t& Range() { return range; }

  //! Access the value of k.
  const size_t& K() const { return k; }
  //! Modify the value of k.
  size_t K() { return k; }

  //! Get the optimizer.
  const OptimizerType& Optimizer() const { return optimizer; }
  OptimizerType& Optimizer() { return optimizer; }

 private:
  //! Dataset reference.
  const arma::mat& dataset;

  //! Labels reference.
  const arma::Row<size_t>& labels;

  //! Number of target points.
  size_t k;

  //! Regularization value.
  double regularization;

  //! Range after which impostors need to be recalculated.
  size_t range;

  //! Metric to be used.
  MetricType metric;

  //! The optimizer to use.
  OptimizerType optimizer;
}; // class LMNN

} // namespace mlpack

// Include the implementation.
#include "lmnn_impl.hpp"

#endif
