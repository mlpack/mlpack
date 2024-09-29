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

template<typename MatType>
DropConnectType<MatType>::DropConnectType() :
    Layer<MatType>(),
    ratio(0.5),
    scale(2.0),
    baseLayer(new LinearType<MatType>(0))
{
  // Nothing to do here.
}

template<typename MatType>
DropConnectType<MatType>::DropConnectType(
    const size_t outSize,
    const double ratio) :
    Layer<MatType>(),
    ratio(ratio),
    scale(1.0 / (1 - ratio)),
    baseLayer(new LinearType<MatType>(outSize))
{
  // Nothing to do.
}

template<typename MatType>
DropConnectType<MatType>::~DropConnectType()
{
  delete baseLayer;
}

template<typename MatType>
DropConnectType<MatType>::DropConnectType(const DropConnectType& other) :
    Layer<MatType>(other),
    ratio(other.ratio),
    scale(other.scale),
    baseLayer(other.baseLayer->Clone())
{
  // Nothing to do.
}

template<typename MatType>
DropConnectType<MatType>::DropConnectType(DropConnectType&& other) :
    Layer<MatType>(std::move(other)),
    ratio(std::move(other.ratio)),
    scale(std::move(other.scale)),
    baseLayer(std::move(other.baseLayer))
{
  // Nothing to do.
}

template<typename MatType>
DropConnectType<MatType>&
DropConnectType<MatType>::operator=(const DropConnectType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    ratio = other.ratio;
    scale = other.scale;
    baseLayer = other.baseLayer->Clone();
  }

  return *this;
}

template<typename MatType>
DropConnectType<MatType>&
DropConnectType<MatType>::operator=(DropConnectType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    ratio = std::move(other.ratio);
    scale = std::move(other.scale);
    baseLayer = std::move(other.baseLayer);
  }

  return *this;
}

template<typename MatType>
void DropConnectType<MatType>::Forward(const MatType& input, MatType& output)
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
    mask.randu(denoise.n_rows, denoise.n_cols);
    mask.transform([&](double val) { return (val > ratio); });

    baseLayer->Parameters() = denoise % mask;
    baseLayer->Forward(input, output);

    output = output * scale;
  }
}

template<typename MatType>
void DropConnectType<MatType>::Backward(
    const MatType& input,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  baseLayer->Backward(input, output, gy, g);
}

template<typename MatType>
void DropConnectType<MatType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  baseLayer->Gradient(input, error, gradient);

  // Denoise the weights.
  baseLayer->Parameters() = denoise;
}

template<typename MatType>
void DropConnectType<MatType>::ComputeOutputDimensions()
{
  // Propagate input dimensions to the base layer.
  baseLayer->InputDimensions() = this->inputDimensions;
  this->outputDimensions = baseLayer->OutputDimensions();
}

template<typename MatType>
void DropConnectType<MatType>::SetWeights(const MatType& weightsIn)
{
  baseLayer->SetWeights(weightsIn);
}

template<typename MatType>
template<typename Archive>
void DropConnectType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(ratio));
  ar(CEREAL_NVP(scale));
  ar(CEREAL_POINTER(baseLayer));
}

}  // namespace mlpack

#endif
