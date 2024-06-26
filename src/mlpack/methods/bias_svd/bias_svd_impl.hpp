/**
 * @file methods/bias_svd/bias_svd_impl.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of Bias SVD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_BIAS_SVD_BIAS_SVD_IMPL_HPP
#define MLPACK_METHODS_BIAS_SVD_BIAS_SVD_IMPL_HPP

namespace mlpack {

template<typename OptimizerType, typename MatType, typename VecType>
BiasSVD<OptimizerType, MatType, VecType>::BiasSVD(const size_t iterations,
                                                  const double alpha,
                                                  const double lambda) :
    iterations(iterations),
    alpha(alpha),
    lambda(lambda)
{
  // Nothing to do.
}

template<typename OptimizerType, typename MatType, typename VecType>
void BiasSVD<OptimizerType, MatType, VecType>::Apply(const MatType& data,
                                                     const size_t rank,
                                                     MatType& u,
                                                     MatType& v,
                                                     VecType& p,
                                                     VecType& q)
{
  // batchSize is 1 in our implementation of Bias SVD.
  // batchSize other than 1 has not been supported yet.
  const int batchSize = 1;
  Log::Warn << "The batch size for optimizing BiasSVD is 1."
      << std::endl;

  // Make the optimizer object using a BiasSVDFunction object.
  BiasSVDFunction<arma::mat> biasSVDFunc(data, rank, lambda);
  ens::StandardSGD optimizer(alpha, batchSize,
      iterations * data.n_cols);

  // Get optimized parameters.
  MatType parameters = biasSVDFunc.GetInitialPoint();
  optimizer.Optimize(biasSVDFunc, parameters);

  // Constants for extracting user and item matrices.
  const size_t numUsers = max(data.row(0)) + 1;
  const size_t numItems = max(data.row(1)) + 1;

  // Extract user and item matrices, user and item bias from the optimized
  // parameters.
  u = parameters.submat(0, numUsers, rank - 1, numUsers + numItems - 1).t();
  v = parameters.submat(0, 0, rank - 1, numUsers - 1);
  p = parameters.row(rank).subvec(numUsers, numUsers + numItems - 1).t();
  q = parameters.row(rank).subvec(0, numUsers - 1).t();
}

} // namespace mlpack

#endif
