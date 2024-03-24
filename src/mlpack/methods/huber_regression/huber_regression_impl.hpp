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
HuberRegressor<MatType, VecType>::HuberRegressor(const MatType& X, const VecType& y,double delta, size_t maxIter, double tol)
    : delta(delta), maxIter(maxIter), tol(tol) {
      Train(X,y);
    }

template<typename MatType, typename VecType>
void HuberRegressor<MatType, VecType>::Train(const MatType& X, const VecType& y) {
  // Initialize coefficients
  coef = VecType(X.n_cols, arma::fill::zeros);

  // Define the Huber loss function
  HuberLoss<MatType, VecType> loss(X, y, delta);

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
