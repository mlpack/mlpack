/**
 * @file methods/huber_regression/huber_regression.hpp
 * @author Anna Sai Nikhil
 *
 * Implementation of Huber Regression
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_HPP
#define MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_HPP

#include <mlpack/core.hpp>

namespace mlpack {

template<typename MatType = arma::mat, typename VecType = arma::vec>
class HuberLoss
{
 public:
  /**
   * Construct the Huber loss with the given data and delta value.
   *
   * @param X Input data (matrix).
   * @param y Responses (vector).
   * @param delta Delta value for Huber loss.
   */
  HuberLoss(const MatType& X, const VecType& y, const double delta);

  /**
   * Evaluate the Huber loss for the given parameters.
   *
   * @param theta Regression coefficients.
   * @return Loss value.
   */
  double Evaluate(const MatType& theta);

  /**
   * Compute the gradient of the Huber loss for the given parameters.
   *
   * @param theta Regression coefficients.
   * @param gradient Output gradient.
   */
  void Gradient(const MatType& theta, MatType& gradient);

 private:
  const MatType& X;    // Input data.
  const VecType& y;    // Responses.
  const double delta;    // Delta value for Huber loss.
};

template<typename MatType = arma::mat, typename VecType = arma::vec>
class HuberRegressor
{
 public:
  /**
   * Construct the Huber Regressor with default parameters.
   *
   * @param epsilon Epsilon value for the Huber Regressor.
   * @param maxIter Maximum number of iterations for optimization.
   * @param tol Tolerance for convergence.
   */
  HuberRegressor(double epsilon = 1.35, size_t maxIter = 100, double tol = 1e-5);

  /**
   * Fit the Huber Regressor to the given data.
   *
   * @param X Input data (matrix).
   * @param y Responses (vector).
   */
  void Train(const MatType& X, const VecType& y);

  /**
   * Predict responses for the given input data.
   *
   * @param X Input data (matrix).
   * @return Predicted responses.
   */
  VecType Predict(const MatType& X) const;

  /**
   * Get the coefficients of the fitted model.
   *
   * @return Coefficients.
   */
  VecType getCoef() const;

 private:
  double epsilon;        // Epsilon value for the Huber Regressor.
  size_t maxIter;        // Maximum number of iterations for optimization.
  double tol;            // Tolerance for convergence.
  VecType coef;          // Coefficients of the model.
};

} // namespace mlpack

// Include implementation.
#include "huber_regression_impl.hpp"

#endif // MLPACK_METHODS_HUBER_REGRESSION_HPP
