/**
 * @file cross_entropy_error_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of the cross-entropy performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "cross_entropy_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
CrossEntropyError<InputDataType, OutputDataType>::CrossEntropyError(double eps)
  : eps(eps)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double CrossEntropyError<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return -arma::accu(target % arma::log(input + eps) +
                     (1. - target) % arma::log(1. - input + eps));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void CrossEntropyError<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output = (1. - target) / (1. - input + eps) - target / (input + eps);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CrossEntropyError<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(eps);
}

} // namespace ann
} // namespace mlpack

#endif
