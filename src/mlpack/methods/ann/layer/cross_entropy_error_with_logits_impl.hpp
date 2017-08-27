/**
 * @file cross_entropy_error_with_logits_impl.hpp
 * @author Kris Singh
 *
 * Implementation of the cross-entropy with logits performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_LOGIT_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_LOGIT_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "cross_entropy_error_with_logits.hpp"
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
CrossEntropyErrorLogits<InputDataType, OutputDataType>
::CrossEntropyErrorLogits()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
double CrossEntropyErrorLogits<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, const arma::Mat<eT>&& target)
{
  eT loss = 0;
  for (size_t i = 0; i < input.n_elem; i++)
    if (input(i) > 0)
    {
      loss += input(i) - input(i) * target(i) +
              SoftplusFunction::Fn(-std::abs(input(i)));
    }
    else
    {
      loss += input(i) * target(i) +
              SoftplusFunction::Fn(-std::abs(input(i)));;
    }

  return loss / input.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void CrossEntropyErrorLogits<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input,
    const arma::Mat<eT>&& target,
    arma::Mat<eT>&& output)
{
  output = input;
  for (size_t i = 0; i < input.n_elem; i++)
  {
    if (input(i) > 0)
      output(i) = 1 - target(i) - SoftplusFunction::Deriv(-std::abs(input(i)));
    else
      output(i) = -(target(i)) + SoftplusFunction::Deriv(-std::abs(input(i)));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CrossEntropyErrorLogits<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here
}

} // namespace ann
} // namespace mlpack

#endif
