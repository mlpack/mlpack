/**
 * @file methods/ann/layer/reinforce_normal_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the ReinforceNormalLayer class, which implements the
 * REINFORCE algorithm for the normal distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_IMPL_HPP

// In case it hasn't yet been included.
#include "reinforce_normal.hpp"

namespace mlpack {

template<typename InputDataType, typename OutputDataType>
ReinforceNormalType<InputDataType, OutputDataType>::ReinforceNormalType(
    const double stdev) : stdev(stdev), reward(0.0), deterministic(false)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void ReinforceNormalType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (!deterministic)
  {
    // Multiply by standard deviations and re-center the means to the mean.
    output.randn(input.n_rows, input.n_cols) * stdev + input;
    moduleInputParameter.push_back(input);
  }
  else
  {
    // Use maximum a posteriori.
    output = input;
  }
}

template<typename InputType, typename OutputType>
void ReinforceNormalType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& /* gy */, OutputType& g)
{
  g = (input - moduleInputParameter.back()) / std::pow(stdev, 2.0);

  // Multiply by reward and multiply by -1.
  g *= reward;
  g *= -1;

  moduleInputParameter.pop_back();
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ReinforceNormalType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(stdev));
}

} // namespace mlpack

#endif
