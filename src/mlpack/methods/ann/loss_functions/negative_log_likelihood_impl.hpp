/**
 * @file methods/ann/loss_functions/negative_log_likelihood_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the NegativeLogLikelihoodType class.
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

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename MatType>
NegativeLogLikelihoodType<MatType>::NegativeLogLikelihoodType(
    const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type
NegativeLogLikelihoodType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  using ElemType = typename MatType::elem_type;

  // Sanity check the inputs.
  Log::Assert(all(vectorise(target >= ElemType(0) &&
                            target < ElemType(prediction.n_rows))),
      "NegativeLogLikelihood::Forward(): labels must be between 0 and "
      "(numClasses - 1).");

  // Assemble the indices in `prediction` we are looking for.
  // For each i, we want to access prediction(target(i), i).
  const ElemType lossSum = -accu(prediction.elem(
      conv_to<typename GetUColType<MatType>::type>::from(
          target + (prediction.n_rows * linspace<MatType>(0,
          prediction.n_cols - 1, prediction.n_cols).t()))));

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void NegativeLogLikelihoodType<MatType>::Backward(
      const MatType& prediction,
      const MatType& target,
      MatType& loss)
{
  using ElemType = typename MatType::elem_type;

  // Sanity check the inputs.
  Log::Assert(all(vectorise(target >= ElemType(0) &&
                            target < ElemType(prediction.n_rows))),
      "NegativeLogLikelihood::Forward(): labels must be between 0 and "
      "(numClasses - 1).");

  loss = zeros<MatType>(prediction.n_rows, prediction.n_cols);
  loss.elem(conv_to<typename GetUColType<MatType>::type>::from(
      target + (prediction.n_rows * linspace<MatType>(0,
      prediction.n_cols - 1, prediction.n_cols).t()))).fill(-1);

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void NegativeLogLikelihoodType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
