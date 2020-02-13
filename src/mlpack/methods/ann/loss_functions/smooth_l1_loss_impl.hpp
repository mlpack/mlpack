/**
 * @file smooth_l1_loss_impl.hpp
 * @author Saksham Rastogi
 *
 * Implementation of the Smooth L1 Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SMOOTH_L1_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SMOOTH_L1_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "smooth_l1_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SmoothL1Loss<InputDataType, OutputDataType>::SmoothL1Loss(const bool takeMean,
                                                          const double sigma) :
    takeMean(takeMean),
    sigma(sigma)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double SmoothL1Loss<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  double output = 0;
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    double delta = input[i] - target[i];
    if (std::abs(delta) < 1/std::pow(sigma,2)) {
      output += 0.5 * std::pow(sigma,2) * std::pow(delta,2);
    } else {
      output += std::abs(delta) - 0.5 / std::pow(delta,2);
    }
  }

  return takeMean ? output / input.n_elem : output;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void SmoothL1Loss<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output = arma::zeros<OutputType>(input.n_rows, input.n_cols);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    double delta = input[i] - target[i];
    if (std::abs(delta) < 1/std::pow(sigma,2)) {
      output[i] = std::pow(sigma,2) * delta;
    } else {
      output[i] = delta > 0 : 1 : -1;
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SmoothL1Loss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
