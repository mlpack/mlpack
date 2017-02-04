/**
 * @file regularized_svd_impl.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_IMPL_HPP
#define MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_IMPL_HPP

namespace mlpack {
namespace svd {

template<template<typename> class OptimizerType>
RegularizedSVD<OptimizerType>::RegularizedSVD(const size_t iterations,
                                              const double alpha,
                                              const double lambda) :
    iterations(iterations),
    alpha(alpha),
    lambda(lambda)
{
  // Nothing to do.
}

template<template<typename> class OptimizerType>
void RegularizedSVD<OptimizerType>::Apply(const arma::mat& data,
                                          const size_t rank,
                                          arma::mat& u,
                                          arma::mat& v)
{
  // Make the optimizer object using a RegularizedSVDFunction object.
  RegularizedSVDFunction rSVDFunc(data, rank, lambda);
  mlpack::optimization::SGD<RegularizedSVDFunction> optimizer(rSVDFunc, alpha,
      iterations * data.n_cols);

  // Get optimized parameters.
  arma::mat parameters = rSVDFunc.GetInitialPoint();
  optimizer.Optimize(parameters);

  // Constants for extracting user and item matrices.
  const size_t numUsers = max(data.row(0)) + 1;
  const size_t numItems = max(data.row(1)) + 1;

  // Extract user and item matrices from the optimized parameters.
  u = parameters.submat(0, numUsers, rank - 1, numUsers + numItems - 1).t();
  v = parameters.submat(0, 0, rank - 1, numUsers - 1);
}

} // namespace svd
} // namespace mlpack

#endif
