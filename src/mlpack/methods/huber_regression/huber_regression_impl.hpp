/**
 * @file methods/huber_regression/huber_regression_impl.hpp
 * @author Anna Sai Nikhil
 *
 * Implementation of robust regression using the Huber loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_IMPL_HPP

#include "huber_regression.hpp"

namespace mlpack {

template<typename MatType, typename VecType>
HuberLoss<MatType, VecType>::HuberLoss(const MatType& X, const VecType& y, const double delta)
    : X(X), y(y), delta(delta) {}

template<typename MatType, typename VecType>
double HuberLoss<MatType, VecType>::Evaluate(const MatType& theta) {
  VecType residuals = y - X * theta;
  VecType huber = arma::zeros<VecType>(residuals.n_elem);
  for (size_t i = 0; i < residuals.n_elem; ++i) {
    if (std::abs(residuals[i]) <= delta) {
      huber[i] = 0.5 * std::pow(residuals[i], 2);
    } else {
      huber[i] = delta * (std::abs(residuals[i]) - 0.5 * delta);
    }
  }
  return arma::accu(huber);
}

template<typename MatType, typename VecType>
void HuberLoss<MatType, VecType>::Gradient(const MatType& theta, MatType& gradient) {
  VecType residuals = y - X * theta;
  gradient = -X.t() * (residuals % arma::clamp(residuals / delta, -1, 1));
}

template<typename MatType, typename VecType>
HuberRegressor<MatType, VecType>::HuberRegressor(double epsilon, size_t maxIter, double tol)
    : epsilon(epsilon), maxIter(maxIter), tol(tol) {}

template<typename MatType, typename VecType>
void HuberRegressor<MatType, VecType>::Train(const MatType& X, const VecType& y) {
  // Initialize coefficients
  coef = VecType(X.n_cols, arma::fill::zeros);

  // Define the Huber loss function
  HuberLoss<MatType, VecType> loss(X, y, epsilon);

  // Define the L-BFGS optimizer
  ens::L_BFGS optimizer(maxIter, X.n_cols * 10, tol);

  // Optimize!
  optimizer.Optimize(loss, coef);
}

template<typename MatType, typename VecType>
VecType HuberRegressor<MatType, VecType>::Predict(const MatType& X) const {
  return X * coef;
}

template<typename MatType, typename VecType>
VecType HuberRegressor<MatType, VecType>::getCoef() const {
  return coef;
}

} // namespace mlpack

#endif // MLPACK_METHODS_HUBER_REGRESSION_HUBER_REGRESSION_IMPL_HPP
