/**
 * @file hinge_embedding_loss_impl.hpp
 * @author Lakshya Ojha
 *
 * Implementation of the Hinge Embedding loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_EMBEDDING_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_EMBEDDING_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "hinge_embedding_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
HingeEmbeddingLoss<InputDataType, OutputDataType>::HingeEmbeddingLoss()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double HingeEmbeddingLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input, const TargetType& target)
{
  return arma::accu(arma::max(1-input % target, 0.));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void HingeEmbeddingLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output = (input < 1 / target) % -target;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void HingeEmbeddingLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
