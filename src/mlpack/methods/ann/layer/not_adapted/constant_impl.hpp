/**
 * @file methods/ann/layer/constant_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Constant class, which outputs a constant value given
 * any input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONSTANT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONSTANT_IMPL_HPP

// In case it hasn't yet been included.
#include "constant.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
ConstantType<InputType, OutputType>::ConstantType() :
    outSize(0)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConstantType<InputType, OutputType>::ConstantType(
    const size_t outSize,
    const double scalar) :
    outSize(outSize)
{
  constantOutput = OutputType(outSize, 1);
  constantOutput.fill(scalar);
}

template<typename InputType, typename OutputType>
ConstantType<InputType, OutputType>::ConstantType(
    const ConstantType<InputType, OutputType>& other) :
    outSize(other.outSize),
    constantOutput(other.constantOutput)
{
  // Nothing else to do.
}

template<typename InputType, typename OutputType>
ConstantType<InputType, OutputType>::ConstantType(
    ConstantType<InputType, OutputType>&& other) :
    outSize(other.outSize),
    constantOutput(std::move(other.constantOutput))
{
  other.outSize = 1;
  other.constantOutput = OutputType(other.outSize, 1);
}

template<typename InputType, typename OutputType>
ConstantType<InputType, OutputType>&
ConstantType<InputType, OutputType>::operator=(
    const ConstantType<InputType, OutputType>& other)
{
  if (this != &other)
  {
    outSize = other.outSize;
    constantOutput = other.constantOutput;
  }

  return *this;
}

template<typename InputType, typename OutputType>
ConstantType<InputType, OutputType>&
ConstantType<InputType, OutputType>::operator=(
    ConstantType<InputType, OutputType>&& other)
{
  if (this != *other)
  {
    outSize = other.outSize;
    constantOutput = std::move(other.constantOutput);

    other.outSize = 1;
    other.constantOutput = OutputType(other.outSize, 1);
  }

  return *this;
}

template<typename InputType, typename OutputType>
void ConstantType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  output = constantOutput;
}

template<typename InputType, typename OutputType>
void ConstantType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& /* gy */, OutputType& g)
{
  g.zeros();
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ConstantType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(constantOutput));
  if (Archive::is_loading::value)
    outSize = constantOutput.n_elem;
}

} // namespace mlpack

#endif
