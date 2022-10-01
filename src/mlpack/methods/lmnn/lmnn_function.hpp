/**
 * @file methods/lmnn/lmnn_function.hpp
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

#include "constraints.hpp"

namespace mlpack {

/**
 * The Large Margin Nearest Neighbors function.
 *
 * The actual function is
 *
 * @f$ \epsilon(M) = \sum_{ij}\eta_{ij}|| L x_i - L x_j ||^2 +
 *     c\sum_{ijl}\eta_{ij}(1-y_{il})[1 + || L x_i - L x_j ||^2 -
 *     || L x_i - L x_l ||^2)]_{+} @f$
 *
 * where x_n represents a point and A is the current scaling matrix.
 *
 * This class is more flexible than the original paper, allowing an arbitrary
 * metric function to be used in place of || A x_i - A x_j ||^2, meaning that
 * the squared Euclidean distance is not the only allowed metric for LMNN.
 * However, that is probably the best way to use this class.
 *
 * In addition to the standard Evaluate() and Gradient() functions which mlpack
 * optimizers use, overloads of Evaluate() and Gradient() are given which only
 * operate on one point in the dataset.  This is useful for optimizers like
 * stochastic gradient descent (see ens::SGD).
 */
template<typename MetricType = SquaredEuclideanDistance>
class LMNNFunction
{
 public:
  /**
   * Constructor for LMNNFunction class.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of target neighbors to be used.
   * @param regularization Regularization value.
   * @param range Range after which impostors need to be recalculated.
   * @param metric Type of metric used for computation.
   */
  LMNNFunction(const arma::mat& dataset,
               const arma::Row<size_t>& labels,
               size_t k,
               double regularization,
               size_t range,
               MetricType metric = MetricType());


  /**
   * Shuffle the points in the dataset. This may be used by optimizers.
   */
  void Shuffle();

  /**
   * Evaluate the LMNN function for the given transformation matrix.  This is the
   * non-separable implementation, where the objective function is not
   * decomposed into the sum of several objective functions.
   *
   * @param transformation Transformation matrix of Mahalanobis distance.
   */
  double Evaluate(const arma::mat& transformation);

  /**
   * Evaluate the LMNN objective function for the given transformation matrix on
   * the given batch size from a given inital point of the dataset.
   * This is the separable implementation, where the objective
   * function is decomposed into the sum of many objective
   * functions, and here, only one of those constituent objective functions is
   * returned.
   *
   * @param transformation Transformation matrix of Mahalanobis distance.
   * @param begin Index of the initial point to use for objective function.
   * @param batchSize Number of points to use for objective function.
   */
  double Evaluate(const arma::mat& transformation,
                  const size_t begin,
                  const size_t batchSize = 1);

  /**
   * Evaluate the gradient of the LMNN function for the given transformation
   * matrix.  This is the non-separable implementation, where the objective
   * function is not decomposed into the sum of several objective functions.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param transformation Transformation matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  template<typename GradType>
  void Gradient(const arma::mat& transformation, GradType& gradient);

  /**
   * Evaluate the gradient of the LMNN function for the given transformation
   * matrix on the given batch size, from a given initial point of the dataset.
   * This is the separable implementation, where the objective function is
   * decomposed into the sum of many objective functions, and here,
   * only one of those constituent objective functions is returned.
   * The type of the gradient parameter is a template
   * argument to allow the computation of a sparse gradient.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param transformation Transformation matrix of Mahalanobis distance.
   * @param begin Index of the initial point to use for objective function.
   * @param gradient Matrix to store the calculated gradient in.
   * @param batchSize Number of points to use for objective function.
   */
  template<typename GradType>
  void Gradient(const arma::mat& transformation,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize = 1);

  /**
   * Evaluate the LMNN objective function together with gradient for the given
   * transformation matrix.  This is the non-separable implementation, where the
   * objective function is not decomposed into the sum of several objective
   * functions.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param transformation Transformation matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& transformation,
                              GradType& gradient);

  /**
   * Evaluate the LMNN objective function together with gradient for the given
   * transformation matrix on the given batch size, from a given initial point of
   * the dataset. This is the separable implementation, where the objective
   * function is decomposed into the sum of many objective functions, and
   * here, only one of those constituent objective functions is returned.
   * The type of the gradient parameter is a template
   * argument to allow the computation of a sparse gradient.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param transformation Transformation matrix of Mahalanobis distance.
   * @param begin Index of the initial point to use for objective function.
   * @param gradient Matrix to store the calculated gradient in.
   * @param batchSize Number of points to use for objective function.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& transformation,
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

  //! Access the value of range.
  const size_t& Range() const { return range; }
  //! Modify the value of k.
  size_t& Range() { return range; }

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
  //! Cache distance. Used to avoid repetive calculation.
  arma::mat distance;
  //! Number of target neighbors.
  size_t k;
  //! The instantiated metric.
  MetricType metric;
  //! Regularization value.
  double regularization;
  //! Keep iterations count.
  size_t iteration;
  //! Range after which impostors need to be recalculated.
  size_t range;
  //! Constraints Object.
  Constraints<MetricType> constraint;
  //! Holds pre-calculated cij.
  arma::mat pCij;
  //! Holds the norm of each data point.
  arma::vec norm;
  //! Hold previous eval values for each datapoint.
  arma::cube evalOld;
  //! Hold previous maximum norm of impostor.
  arma::mat maxImpNorm;
  //! Holds previous transformation matrix. Used for L-BFGS like optimizer.
  arma::mat transformationOld;
  //! Holds previous transformation matrices.
  std::vector<arma::mat> oldTransformationMatrices;
  //! Holds number of points which are using each transformation matrix.
  std::vector<size_t> oldTransformationCounts;
  //! Holds points to transformation matrix mapping.
  arma::vec lastTransformationIndices;
  //! Used for storing points to re-calculate impostors for.
  arma::uvec points;
  //! Flag for controlling use of bounds over impostors.
  bool impBounds;
  /**
  * Precalculate the gradient part due to target neighbors and stores
  * the result as a matrix. Used for L-BFGS like optimizers which does not
  * uses batches.
  */
  inline void Precalculate();
  //! Update cache transformation matrices.
  inline void UpdateCache(const arma::mat& transformation,
                          const size_t begin,
                          const size_t batchSize);
  //! Calculate norm of change in transformation.
  inline void TransDiff(std::map<size_t, double>& transformationDiffs,
                        const arma::mat& transformation,
                        const size_t begin,
                        const size_t batchSize);
};

} // namespace mlpack

#include "lmnn_function_impl.hpp"

#endif
