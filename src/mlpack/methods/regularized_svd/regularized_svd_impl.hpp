/**
 * @file regularized_svd_impl.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
