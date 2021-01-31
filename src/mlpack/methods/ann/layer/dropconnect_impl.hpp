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
    ratio(0.5),
    scale(2.0),
    deterministic(true)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
DropConnectType<InputType, OutputType>::DropConnectType(
    const size_t inSize,
    const size_t outSize,
    const double ratio) :
    ratio(ratio),
    scale(1.0 / (1 - ratio)),
    deterministic(false),
    baseLayer(new LinearType<InputType, OutputType>(inSize, outSize))
{
  network.push_back(baseLayer);
}

template<typename InputType, typename OutputType>
void DropConnectType<InputType, OutputType>::Forward(
    const InputType& input,
    OutputType& output)
{
  // The DropConnect mask will not be multiplied in the deterministic mode
  // (during testing).
  if (deterministic)
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
    OutputType& /* gradient */)
{
  baseLayer->Gradient(input, error, baseLayer->Gradient());

  // Denoise the weights.
  baseLayer->Parameters() = denoise;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void DropConnectType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // Delete the old network first, if needed.
  if (cereal::is_loading<Archive>())
    delete baseLayer;

  ar(CEREAL_NVP(ratio));
  ar(CEREAL_NVP(scale));
  ar(CEREAL_VARIANT_POINTER(baseLayer));

  if (cereal::is_loading<Archive>())
  {
    network.clear();
    network.push_back(baseLayer);
  }
}

}  // namespace ann
}  // namespace mlpack

#endif
