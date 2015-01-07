/**
 * @file regularized_svd_impl.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_IMPL_HPP
#define __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_IMPL_HPP

namespace mlpack {
namespace svd {

template<template<typename> class OptimizerType>
RegularizedSVD<OptimizerType>::RegularizedSVD(const arma::mat& data,
                                              arma::mat& u,
                                              arma::mat& v,
                                              const size_t rank,
                                              const size_t iterations,
                                              const double alpha,
                                              const double lambda) :
    data(data),
    rank(rank),
    iterations(iterations),
    alpha(alpha),
    lambda(lambda),
    rSVDFunc(data, rank, lambda),
    optimizer(rSVDFunc, alpha, iterations * data.n_cols)
{
  arma::mat parameters = rSVDFunc.GetInitialPoint();

  // Train the model.
  Timer::Start("regularized_svd_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("regularized_svd_optimization");
  
  const size_t numUsers = max(data.row(0)) + 1;
  const size_t numItems = max(data.row(1)) + 1;
  
  u = parameters.submat(0, 0, rank - 1, numUsers - 1);
  v = parameters.submat(0, numUsers, rank - 1, numUsers + numItems - 1);
}

}; // namespace svd
}; // namespace mlpack

#endif
