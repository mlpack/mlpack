/**
 * @file methods/svdplusplus/svdplusplus_impl.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of SVD++.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_IMPL_HPP
#define MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_IMPL_HPP

namespace mlpack {

template<typename OptimizerType>
SVDPlusPlus<OptimizerType>::SVDPlusPlus(const size_t iterations,
                                        const double alpha,
                                        const double lambda) :
    iterations(iterations),
    alpha(alpha),
    lambda(lambda)
{
  // Nothing to do.
}

template<typename OptimizerType>
void SVDPlusPlus<OptimizerType>::Apply(const arma::mat& data,
                                       const arma::mat& implicitData,
                                       const size_t rank,
                                       arma::mat& u,
                                       arma::mat& v,
                                       arma::vec& p,
                                       arma::vec& q,
                                       arma::mat& y)
{
  // batchSize is 1 in our implementation of SVDPlusPlus.
  // batchSize other than 1 has not been supported yet.
  const int batchSize = 1;
  Log::Warn << "The batch size for optimizing SVDPlusPlus is 1."
      << std::endl;

  // Converts implicitData to the form of sparse matrix.
  arma::sp_mat cleanedData;
  CleanData(implicitData, cleanedData, data);

  // Make the optimizer object using a SVDPlusPlusFunction object.
  SVDPlusPlusFunction<arma::mat> svdPPFunc(data, cleanedData, rank, lambda);
  ens::StandardSGD optimizer(alpha, batchSize,
      iterations * data.n_cols);

  // Get optimized parameters.
  arma::mat parameters = svdPPFunc.GetInitialPoint();
  optimizer.Optimize(svdPPFunc, parameters);

  // Constants for extracting user and item matrices.
  const size_t numUsers = max(data.row(0)) + 1;
  const size_t numItems = max(data.row(1)) + 1;

  // Extract user and item matrices, user and item bias, item implicit matrix
  // from the optimized parameters.
  u = parameters.submat(0, numUsers, rank - 1, numUsers + numItems - 1).t();
  v = parameters.submat(0, 0, rank - 1, numUsers - 1);
  p = parameters.row(rank).subvec(numUsers, numUsers + numItems - 1).t();
  q = parameters.row(rank).subvec(0, numUsers - 1).t();
  y = parameters.submat(0, numUsers + numItems, rank - 1,
      numUsers + 2 * numItems - 1);
}

// Use whether a user rates an item as binary implicit data when implicitData
// is not given.
template<typename OptimizerType>
void SVDPlusPlus<OptimizerType>::Apply(const arma::mat& data,
                                       const size_t rank,
                                       arma::mat& u,
                                       arma::mat& v,
                                       arma::vec& p,
                                       arma::vec& q,
                                       arma::mat& y)
{
  arma::mat implicitData = data.submat(0, 0, 1, data.n_cols - 1);
  Apply(data, implicitData, rank, u, v, p, q, y);
}

template<typename OptimizerType>
void SVDPlusPlus<OptimizerType>::CleanData(const arma::mat& implicitData,
                                           arma::sp_mat& cleanedData,
                                           const arma::mat& data)
{
  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, implicitData.n_cols);
  arma::vec values(implicitData.n_cols);
  for (size_t i = 0; i < implicitData.n_cols; ++i)
  {
    // We have to transpose it because items are rows, and users are columns.
    locations(1, i) = ((arma::uword) implicitData(0, i));
    locations(0, i) = ((arma::uword) implicitData(1, i));
    values(i) = 1;
  }

  // Find maximum user and item IDs.
  const size_t maxItemID = (size_t) max(data.row(1)) + 1;
  const size_t maxUserID = (size_t) max(data.row(0)) + 1;

  // Fill sparse matrix.
  cleanedData = arma::sp_mat(locations, values, maxItemID, maxUserID);
}

} // namespace mlpack

#endif
