/**
 * @file methods/ann/layer/dropconnect_impl.hpp
 * @author Palash Ahuja
 * @author Marcus Edel
 *
 * Implementation of the DropConnect class, which implements a regularizer
 * that randomly sets connections to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPCONNECT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPCONNECT_IMPL_HPP

// In case it hasn't yet been included.
#include "dropconnect.hpp"

#include "linear.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
DropConnectType<InputType, OutputType>::DropConnectType() :
    Layer<InputType, OutputType>(),
    ratio(0.5),
    scale(2.0),
    baseLayer(new LinearType<InputType, OutputType>(0))
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
DropConnectType<InputType, OutputType>::DropConnectType(
    const size_t outSize,
    const double ratio) :
    Layer<InputType, OutputType>(),
    ratio(ratio),
    scale(1.0 / (1 - ratio)),
    baseLayer(new LinearType<InputType, OutputType>(outSize))
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
DropConnectType<InputType, OutputType>::~DropConnectType()
{
  delete baseLayer;
}

template<typename InputType, typename OutputType>
void DropConnectType<InputType, OutputType>::Forward(
    const InputType& input,
    OutputType& output)
{
  // The DropConnect mask will not be multiplied in testing mode.
  if (!this->training)
  {
    baseLayer->Forward(input, output);
  }
  else
  {
    // Save weights for denoising.
    denoise = baseLayer->Parameters();

    // Scale with input / (1 - ratio) and set values to zero with
    // probability ratio.
    mask = arma::randu<OutputType>(denoise.n_rows, denoise.n_cols);
    mask.transform([&](double val) { return (val > ratio); });

    baseLayer->Parameters() = denoise % mask;
    baseLayer->Forward(input, output);

    output = output * scale;
  }
}

template<typename InputType, typename OutputType>
void DropConnectType<InputType, OutputType>::Backward(
    const InputType& input,
    const OutputType& gy,
    OutputType& g)
{
  baseLayer->Backward(input, gy, g);
}

template<typename InputType, typename OutputType>
void DropConnectType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  baseLayer->Gradient(input, error, gradient);

  // Denoise the weights.
  baseLayer->Parameters() = denoise;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void DropConnectType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(ratio));
  ar(CEREAL_NVP(scale));
  ar(CEREAL_POINTER(baseLayer));
}

}  // namespace ann
}  // namespace mlpack

#endif
