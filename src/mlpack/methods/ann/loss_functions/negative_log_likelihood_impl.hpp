/**
 * @file methods/ann/loss_functions/negative_log_likelihood_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the NegativeLogLikelihood class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP

// In case it hasn't yet been included.
#include "negative_log_likelihood.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename MatType>
NegativeLogLikelihood<MatType>::NegativeLogLikelihood()
{
  // Nothing to do here.
}

template<typename MatType>
double NegativeLogLikelihood<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  double output = 0;
  for (size_t i = 0; i < prediction.n_cols; ++i)
  {
    Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
        "Target class out of range.");

    output -= prediction(target(i), i);
  }

  return output;
}

template<typename MatType>
void NegativeLogLikelihood<MatType>::Backward(
      const MatType& prediction,
      const MatType& target,
      MatType& loss)
{
  loss = arma::zeros<MatType>(prediction.n_rows, prediction.n_cols);
  for (size_t i = 0; i < prediction.n_cols; ++i)
  {
    Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
        "Target class out of range.");

    loss(target(i), i) = -1;
  }
}

} // namespace ann
} // namespace mlpack

#endif
