/**
 * @file methods/logistic_regression/logistic_regression_function.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the logistic regression function, which is meant to be
 * optimized by a separate optimizer class that takes LogisticRegressionFunction
 * as its FunctionType class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/make_alias.hpp>
#include <mlpack/core/math/shuffle_data.hpp>

namespace mlpack {

/**
 * The log-likelihood function for the logistic regression objective function.
 * This is used by various mlpack optimizers to train a logistic regression
 * model.
 */
template<typename MatType = arma::mat>
class LogisticRegressionFunction
{
 public:
  /**
   * Creates the LogisticRegressionFunction.
   *
   * @param predictors The matrix of data points.
   * @param responses The measured data for each point in predictors.
   * @param lambda Regularization constant for ridge regression.
   */
  LogisticRegressionFunction(const MatType& predictors,
                             const arma::Row<size_t>& responses,
                             const double lambda = 0);

  //! Return the regularization parameter (lambda).
  const double& Lambda() const { return lambda; }
  //! Modify the regularization parameter (lambda).
  double& Lambda() { return lambda; }

  //! Return the matrix of predictors.
  const MatType& Predictors() const { return predictors; }
  //! Return the vector of responses.
  const arma::Row<size_t>& Responses() const { return responses; }

  /**
  * Shuffle the order of function visitation.  This may be called by the
  * optimizer.
  */
  void Shuffle();

  /**
   * Evaluate the logistic regression log-likelihood function with the given
   * parameters. Note that if a point has 0 probability of being classified
   * directly with the given parameters, then Evaluate() will return nan (this
   * is kind of a corner case and should not happen for reasonable models).
   *
   * The optimum (minimum) of this function is 0.0, and occurs when each point
   * is classified correctly with very high probability.
   *
   * @param parameters Vector of logistic regression parameters.
   */
  template<typename CoordinatesType>
  typename CoordinatesType::elem_type Evaluate(
      const CoordinatesType& parameters) const;

  /**
   * Evaluate the logistic regression log-likelihood function with the given
   * parameters using the given batch size from the given point index. This is
   * useful for optimizers such as SGD, which require a separable objective
   * function. Note that if the points have 0 probability of being classified
   * correctly with the given parameters, then Evaluate() will return nan (this
   * is kind of a corner case and should not happen for reasonable models).
   *
   * The optimum (minimum) of this function is 0.0, and occurs when the points are
   * classified correctly with very high probability.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param begin Index of the starting point to use for objective function
   *     evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *     objective function evaluation.
   */
  template<typename CoordinatesType>
  typename CoordinatesType::elem_type Evaluate(
      const CoordinatesType& parameters,
      const size_t begin,
      const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param gradient Vector to output gradient into.
   */
  template<typename CoordinatesType, typename GradType>
  void Gradient(const CoordinatesType& parameters, GradType& gradient) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters, for the given batch size from a given point in
   * the dataset. This is useful for optimizers such as SGD, which require a
   * separable objective function.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param begin Index of the starting point to use for objective function
   *     gradient evaluation.
   * @param gradient Vector to output gradient into.
   * @param batchSize Number of points to be processed as a batch for objective
   *     function gradient evaluation.
   */
  template<typename CoordinatesType, typename GradType>
  void Gradient(const CoordinatesType& parameters,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters, and with respect to only one feature in the
   * dataset. This is useful for optimizers such as SCD, which require
   * partial gradients.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param j Index of the feature with respect to which the gradient is to
   *    be computed.
   * @param gradient Matrix to output gradient into.
   */
  template<typename CoordinatesType, typename GradType>
  void PartialGradient(const CoordinatesType& parameters,
                       const size_t j,
                       GradType& gradient) const;

  /**
   * Evaluate the objective function and gradient of the logistic regression
   * log-likelihood function simultaneously with the given parameters.
   */
  template<typename CoordinatesType, typename GradType>
  typename CoordinatesType::elem_type EvaluateWithGradient(
      const CoordinatesType& parameters,
      GradType& gradient) const;

  /**
   * Evaluate the objective function and gradient of the logistic regression
   * log-likelihood function simultaneously with the given parameters, for
   * the given batch size from a given point in the dataset.
   */
  template<typename CoordinatesType, typename GradType>
  typename CoordinatesType::elem_type EvaluateWithGradient(
      const CoordinatesType& parameters,
      const size_t begin,
      GradType& gradient,
      const size_t batchSize = 1) const;

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return predictors.n_cols; }

  //! Return the number of features(add 1 for the intercept term).
  size_t NumFeatures() const { return predictors.n_rows + 1; }

 private:
  //! The matrix of data points (predictors).  This is an alias until shuffling
  //! is done.
  MatType predictors;
  //! The vector of responses to the input data points.  This is an alias until
  //! shuffling is done.
  arma::Row<size_t> responses;
  //! The regularization parameter for L2-regularization.
  double lambda;
};

} // namespace mlpack

// Include implementation.
#include "logistic_regression_function_impl.hpp"

#endif // MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
