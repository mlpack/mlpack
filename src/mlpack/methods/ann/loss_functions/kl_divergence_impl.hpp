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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
KLDivergence<InputDataType, OutputDataType>::KLDivergence(const bool takeMean) :
    takeMean(takeMean)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
KLDivergence<InputDataType, OutputDataType>::Forward(const InputType& input,
                                                     const TargetType& target)
{
  if (takeMean)
  {
    return arma::as_scalar(arma::mean(
        arma::mean(input % (arma::log(input) - arma::log(target)))));
  }
  else
  {
    return arma::accu(input % (arma::log(input) - arma::log(target)));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void KLDivergence<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  if (takeMean)
  {
    output = arma::mean(arma::mean(arma::log(input) - arma::log(target) + 1));
  }
  else
  {
    output = arma::accu(arma::log(input) - arma::log(target) + 1);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void KLDivergence<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(takeMean);
}

} // namespace ann
} // namespace mlpack

#endif
