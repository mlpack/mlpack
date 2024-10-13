/**
 * @file methods/ann/loss_functions/cross_entropy_loss_impl.hpp
 * @author Antoine Dubois
 *
 * Implementation of the cross-entropy performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "cross_entropy_loss.hpp"

namespace mlpack {

template<typename MatType>
CrossEntropyLossType<MatType>::CrossEntropyLossType(
    const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type CrossEntropyLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  typedef typename MatType::elem_type ElemType;

  ElemType lossSum = -1.0 * arma::accu(target % prediction); 

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void CrossEntropyLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss = -target;
  if (!reduction)
    loss /= target.n_elem;
}

template<typename MatType>
template<typename Archive>
void CrossEntropyLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif

