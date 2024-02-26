/**
 * @file methods/huber_regression/huber_regression_function.hpp
 * @author Anna Sai Nikhil
 *
 * Implementation of Huber Regression
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_FUNCTION_HPP
#define MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_FUNCTION_HPP

namespace mlpack{
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
}
#include "huber_regression_function_impl.hpp"
#endif
