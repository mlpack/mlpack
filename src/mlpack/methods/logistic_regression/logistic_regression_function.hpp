/**
 * @file logistic_regression_function.hpp
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

#include <mlpack/core.hpp>

namespace mlpack {
namespace regression {

/**
 * The log-likelihood function for the logistic regression objective function.
 * This is used by various mlpack optimizers to train a logistic regression
 * model.
 */
template<typename MatType = arma::mat>
class LogisticRegressionFunction
{
 public:
  LogisticRegressionFunction(const MatType& predictors,
                             const arma::Row<size_t>& responses,
                             const double lambda = 0);

  LogisticRegressionFunction(const MatType& predictors,
                             const arma::Row<size_t>& responses,
                             const arma::vec& initialPoint,
                             const double lambda = 0);

  //! Return the initial point for the optimization.
  const arma::mat& InitialPoint() const { return initialPoint; }
  //! Modify the initial point for the optimization.
  arma::mat& InitialPoint() { return initialPoint; }

  //! Return the regularization parameter (lambda).
  const double& Lambda() const { return lambda; }
  //! Modify the regularization parameter (lambda).
  double& Lambda() { return lambda; }

  //! Return the matrix of predictors.
  const MatType& Predictors() const { return predictors; }
  //! Return the vector of responses.
  const arma::vec& Responses() const { return responses; }

  /**
   * Evaluate the logistic regression log-likelihood function with the given
   * parameters.  Note that if a point has 0 probability of being classified
   * directly with the given parameters, then Evaluate() will return nan (this
   * is kind of a corner case and should not happen for reasonable models).
   *
   * The optimum (minimum) of this function is 0.0, and occurs when each point
   * is classified correctly with very high probability.
   *
   * @param parameters Vector of logistic regression parameters.
   */
  double Evaluate(const arma::mat& parameters) const;

  /**
   * Evaluate the logistic regression log-likelihood function with the given
   * parameters, but using only one data point.  This is useful for optimizers
   * such as SGD, which require a separable objective function.  Note that if
   * the point has 0 probability of being classified correctly with the given
   * parameters, then Evaluate() will return nan (this is kind of a corner case
   * and should not happen for reasonable models).
   *
   * The optimum (minimum) of this function is 0.0, and occurs when the point is
   * classified correctly with very high probability.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param i Index of point to use for objective function evaluation.
   */
  double Evaluate(const arma::mat& parameters, const size_t i) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param gradient Vector to output gradient into.
   */
  void Gradient(const arma::mat& parameters, arma::mat& gradient) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters, and with respect to only one point in the
   * dataset.  This is useful for optimizers such as SGD, which require a
   * separable objective function.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param i Index of points to use for objective function gradient evaluation.
   * @param gradient Vector to output gradient into.
   */
  void Gradient(const arma::mat& parameters,
                const size_t i,
                arma::mat& gradient) const;

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return predictors.n_cols; }

 private:
  //! The initial point, from which to start the optimization.
  arma::mat initialPoint;
  //! The matrix of data points (predictors).
  const MatType& predictors;
  //! The vector of responses to the input data points.
  const arma::Row<size_t>& responses;
  //! The regularization parameter for L2-regularization.
  double lambda;
};

} // namespace regression
} // namespace mlpack

// Include implementation.
#include "logistic_regression_function_impl.hpp"

#endif // MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
