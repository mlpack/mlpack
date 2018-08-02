/**
 * @file boostmetric.hpp
 * @author Manish Kumar
 *
 * Declaration of BoostMetric class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BOOSTMETRIC_BOOSTMETRIC_HPP
#define MLPACK_METHODS_BOOSTMETRIC_BOOSTMETRIC_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/lmnn/constraints.hpp>

namespace mlpack {
namespace boostmetric /** BoostMetric Distance Learning. */ {

/**
 * An implementation of BoostMetric Distance Learning technique.
 * The method seeks to improve clustering & classification algorithms on
 * a dataset by transforming the dataset representation in a more convenient
 * form for them. It introduces the concept of target neighbors and impostors,
 * focusing on the idea that the distance between impostors and the perimeters
 * established by target neighbors should be large and that between target
 * neighbors and data point should be small. It uses an adaptation of
 * coordinate descent optimization as a boosting technique.
 *
 * For more details, see the following published paper:
 *
 * @code
 * @incollection{NIPS2009_3658,
 *   title = {Positive Semidefinite Metric Learning with Boosting},
 *   author = {Shen, Chunhua and Junae Kim and Wang, Lei and Anton Hengel},
 *   booktitle = {Advances in Neural Information Processing Systems 22},
 *   editor = {Y. Bengio and D. Schuurmans and J. D. Lafferty and C. K. I.
 *       Williams and A. Culotta},
 *   pages = {1651--1659},
 *   year = {2009},
 *   publisher = {Curran Associates, Inc.},
 *   url = {http://papers.nips.cc/paper/
 *       3658-positive-semidefinite-metric-learning-with-boosting.pdf}
 * }
 * @endcode
 *
 * @tparam MetricType The type of metric to use for computation.
 */
template<typename MetricType = metric::SquaredEuclideanDistance>
class BOOSTMETRIC
{
 public:
  /**
   * Initialize the BOOSTMETRIC object, passing a dataset (distance metric
   * is learned using this dataset) and labels. Initialization will copy
   * both dataset and labels matrices to internal copies.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of targets to consider.
   */
  BOOSTMETRIC(const arma::mat& dataset,
       const arma::Row<size_t>& labels,
       const size_t k);


  /**
   * Perform boost metric learning. The output distance matrix is written
   * into the passed reference. If the LearnDistance() is called with an
   * outputMatrix with correct dimensions, then that matrix will be used
   * as the starting point for optimization.
   *
   * @param outputMatrix learned distance matrix.
   */
  void LearnDistance(arma::mat& outputMatrix);

  //! Get the dataset reference.
  const arma::mat& Dataset() const { return dataset; }

  //! Get the labels reference.
  const arma::Row<size_t>& Labels() const { return labels; }

  //! Access the value of k.
  const size_t& K() const { return k; }
  //! Modify the value of k.
  size_t& K() { return k; }

  //! Access the tolerance value.
  const size_t& MaxIterations() const { return maxIter; }
  //! Modify the tolerance value.
  size_t& MaxIterations() { return maxIter; }

  //! Access the tolerance value.
  const double& Tolerance() const { return tolerance; }
  //! Modify the tolerance value.
  double& Tolerance() { return tolerance; }

  //! Access the weight tolerance value.
  const double& WTolerance() const { return wTolerance; }
  //! Modify the weight tolerance value.
  double& WTolerance() { return wTolerance; }

  //! Access the upper bound for weight approximation.
  const double& WHigh() const { return wHigh; }
  //! Modify the upper bound for weight approximation.
  double& WHigh() { return wHigh; }

  //! Access the lower bound for weight approximation.
  const double& WLow() const { return wLow; }
  //! Modify the lower bound for weight approximation.
  double& WLow() { return wLow; }

 private:
  //! Dataset reference.
  const arma::mat& dataset;

  //! Labels reference.
  const arma::Row<size_t>& labels;

  //! Number of target points.
  size_t k;

  //! Maximum number of iterations.
  size_t maxIter;

  //! objective tolerance value.
  double tolerance;

  //! binary search approximation tolerance.
  double wTolerance;

  //! upper bound for weight.
  double wHigh;

  //! lower bound for weight.
  double wLow;

  //! Constraints instance.
  lmnn::Constraints<MetricType> constraint;
}; // class BOOSTMETRIC

} // namespace boostmetric
} // namespace mlpack

// Include the implementation.
#include "boostmetric_impl.hpp"

#endif
