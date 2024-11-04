/**
 * @file methods/nca/nca_softmax_error_function.hpp
 * @author Ryan Curtin
 *
 * Implementation of the stochastic neighbor assignment probability error
 * function (the "softmax error").
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_HPP
#define MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/math/make_alias.hpp>
#include <mlpack/core/math/shuffle_data.hpp>

namespace mlpack {

/**
 * The "softmax" stochastic neighbor assignment probability function.
 *
 * The actual function is
 *
 * p_ij = (exp(-|| A x_i - A x_j || ^ 2)) /
 *     (sum_{k != i} (exp(-|| A x_i - A x_k || ^ 2)))
 *
 * where x_n represents a point and A is the current scaling matrix.
 *
 * This class is more flexible than the original paper, allowing an arbitrary
 * metric function to be used in place of || A x_i - A x_j ||^2, meaning that
 * the squared Euclidean distance is not the only allowed metric for NCA.
 * However, that is probably the best way to use this class.
 *
 * In addition to the standard Evaluate() and Gradient() functions which mlpack
 * optimizers use, overloads of Evaluate() and Gradient() are given which only
 * operate on one point in the dataset.  This is useful for optimizers like
 * stochastic gradient descent (see mlpack::optimization::SGD).
 */
template<typename MatType = arma::mat,
         typename LabelsType = arma::Row<size_t>,
         typename DistanceType = SquaredEuclideanDistance>
class SoftmaxErrorFunction
{
 public:
  // Convenience typedef for element type of data.
  using ElemType = typename MatType::elem_type;
  // Convenience typedef for column vector of data.
  using VecType = typename GetColType<MatType>::type;

  /**
   * Initialize with the given kernel; useful when the kernel has some state to
   * store, which is set elsewhere.  If no kernel is given, an empty kernel is
   * used; this way, you can call the constructor with no arguments.  A
   * reference to the dataset we will be optimizing over is also required.
   *
   * @param dataset Matrix containing the dataset.
   * @param labels Vector of class labels for each point in the dataset.
   * @param metric Instantiated metric (optional).
   */
  SoftmaxErrorFunction(const MatType& dataset,
                       const LabelsType& labels,
                       DistanceType metric = DistanceType());

  /**
   * Shuffle the dataset.
   */
  void Shuffle();

  /**
   * Evaluate the softmax function for the given covariance matrix.  This is the
   * non-separable implementation, where the objective function is not
   * decomposed into the sum of several objective functions.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   */
  ElemType Evaluate(const MatType& covariance);

  /**
   * Evaluate the softmax objective function for the given covariance matrix on
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
  ElemType Evaluate(const MatType& covariance,
                    const size_t begin,
                    const size_t batchSize = 1);

  /**
   * Evaluate the gradient of the softmax function for the given covariance
   * matrix.  This is the non-separable implementation, where the objective
   * function is not decomposed into the sum of several objective functions.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  void Gradient(const MatType& covariance, MatType& gradient);

  /**
   * Evaluate the gradient of the softmax function for the given covariance
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
  template <typename GradType>
  void Gradient(const MatType& covariance,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize = 1);

  /**
   * Get the initial point.
   */
  const MatType GetInitialPoint() const;

  /**
   * Get the number of functions the objective function can be decomposed into.
   * This is just the number of points in the dataset.
   */
  size_t NumFunctions() const { return dataset.n_cols; }

 private:
  //! The dataset.  This is an alias until Shuffle() is called.
  MatType dataset;
  //! Labels for each point in the dataset.  This is an alias until Shuffle() is
  //! called.
  LabelsType labels;

  //! The instantiated metric.
  DistanceType distance;

  //! Last coordinates.  Used for the non-separable Evaluate() and Gradient().
  MatType lastCoordinates;
  //! Stretched dataset.  Kept internal to avoid memory reallocations.
  MatType stretchedDataset;
  //! Holds calculated p_i, for the non-separable Evaluate() and Gradient().
  VecType p;
  //! Holds denominators for calculation of p_ij, for the non-separable
  //! Evaluate() and Gradient().
  VecType denominators;

  //! False if nothing has ever been precalculated (only at construction time).
  bool precalculated;

  /**
   * Precalculate the denominators and numerators that will make up the p_ij,
   * but only if the coordinates matrix is different than the last coordinates
   * the Precalculate() method was run with.  This method is only called by the
   * non-separable Evaluate() and Gradient().
   *
   * This will update last_coordinates_ and stretched_dataset_, and also
   * calculate the p_i and denominators_ which are used in the calculation of
   * p_i or p_ij.  The calculation will be O((n * (n + 1)) / 2), which is not
   * great.
   *
   * @param coordinates Coordinates matrix to use for precalculation.
   */
  void Precalculate(const MatType& coordinates);
};

} // namespace mlpack

// Include implementation.
#include "nca_softmax_error_function_impl.hpp"

#endif
