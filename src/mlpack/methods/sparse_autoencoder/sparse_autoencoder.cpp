/**
 * @file methods/sparse_autoencoder/sparse_autoencoder.cpp
 * @author Siddharth Agrawal
 *
 * Implementation of sparse autoencoders.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "sparse_autoencoder.hpp"

namespace mlpack {
namespace nn {

void SparseAutoencoder::GetNewFeatures(arma::mat& data,
                                       arma::mat& features)
{
  const size_t l1 = hiddenSize;
  const size_t l2 = visibleSize;

  Sigmoid(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
      arma::repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
      features);
}

} // namespace nn
} // namespace mlpack
