/**
 * @file reconstruction_loss_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the reconstruction loss performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_RECONSTRUCTION_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_RECONSTRUCTION_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "reconstruction_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
ReconstructionLoss<InputDataType, OutputDataType>::ReconstructionLoss()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double ReconstructionLoss<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  normalDist(input.submat(0, 0, input.n_rows / 2 - 1, input.n_cols - 1),
      input.submat(input.n_rows / 2, 0, input.n_rows - 1, input.n_cols - 1));

  return normalDist.LogProbability(target);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void ReconstructionLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  normalDist.LogProbBackward(target, output);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ReconstructionLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
