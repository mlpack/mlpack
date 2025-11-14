/**
 * @file methods/ann/layer/hard_tanh_impl.hpp
 * @author Dhawal Arora
 * @author Vaibhav Pathak
 *
 * Implementation and implementation of the HardTanH layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_TANH_IMPL_HPP

// In case it hasn't yet been included.
#include "hard_tanh.hpp"

namespace mlpack {

template<typename MatType>
HardTanH<MatType>::HardTanH(
    const double maxValue,
    const double minValue) :
    Layer<MatType>(),
    maxValue(maxValue),
    minValue(minValue)
{
  // Nothing to do here.
}

template<typename MatType>
HardTanH<MatType>::HardTanH(const HardTanH& layer) :
    Layer<MatType>(layer),
    maxValue(layer.maxValue),
    minValue(layer.minValue)
{
  // Nothing to do here.
}

template<typename MatType>
HardTanH<MatType>::HardTanH(HardTanH&& layer) :
    Layer<MatType>(std::move(layer)),
    maxValue(std::move(layer.maxValue)),
    minValue(std::move(layer.minValue))
{
  // Nothing to do here.
}

template<typename MatType>
HardTanH<MatType>& HardTanH<MatType>::operator=(
    const HardTanH& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(layer);
    maxValue = layer.maxValue;
    minValue = layer.minValue;
  }

  return *this;
}

template<typename MatType>
HardTanH<MatType>& HardTanH<MatType>::operator=(HardTanH&& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(std::move(layer));
    maxValue = std::move(layer.maxValue);
    minValue = std::move(layer.minValue);
  }

  return *this;
}
template<typename MatType>
void HardTanH<MatType>::Forward(
    const MatType& input, MatType& output)
{
  #pragma omp parallel for
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output(i) = (input(i) > ElemType(maxValue) ? ElemType(maxValue) :
        (input(i) < ElemType(minValue) ? ElemType(minValue) : input(i)));
  }
}

template<typename MatType>
void HardTanH<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy;

  #pragma omp parallel for
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    // input should not have any values greater than maxValue
    // and lesser than minValue
    if (input(i) <= ElemType(minValue) || input(i) >= ElemType(maxValue))
    {
      g(i) = 0;
    }
  }
}

template<typename MatType>
template<typename Archive>
void HardTanH<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(maxValue));
  ar(CEREAL_NVP(minValue));
}

} // namespace mlpack

#endif
