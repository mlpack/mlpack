/**
 * @file methods/ann/loss_functions/kl_divergence_impl.hpp
 * @author Dakshit Agrawal
 *
 * Implementation of the Kullback–Leibler Divergence error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_IMPL_HPP

// In case it hasn't yet been included.
#include "kl_divergence.hpp"

namespace mlpack {

template<typename MatType>
KLDivergenceType<MatType>::KLDivergenceType(const bool reduction) :
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type KLDivergenceType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType loss = target % (log(target) - prediction);
  typename MatType::elem_type lossSum = accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void KLDivergenceType<MatType>::Backward(
    const MatType& /* prediction */,
    const MatType& target,
    MatType& loss)
{
  loss = -target;

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void KLDivergenceType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
