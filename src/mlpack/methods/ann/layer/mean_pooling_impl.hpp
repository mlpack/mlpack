/**
 * @file methods/ann/layer/mean_pooling_impl.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Implementation of the MeanPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_pooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
MeanPoolingType<InputType, OutputType>::MeanPoolingType()
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
MeanPoolingType<InputType, OutputType>::MeanPoolingType(
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const bool floor) :
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    floor(floor),
    channels(0),
    offset(0),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void MeanPoolingType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  inputTemp = arma::Cube<typename InputType::elem_type>(
      const_cast<InputType&>(input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], batchSize * channels, false, false);

  arma::Cube<typename OutputType::elem_type> outputTemp(output.memptr(),
      outputDimensions[0], outputDimensions[1], batchSize * channels, false,
      true);

  for (size_t s = 0; s < inputTemp.n_slices; s++)
    Pooling(inputTemp.slice(s), outputTemp.slice(s));
}

template<typename InputType, typename OutputType>
void MeanPoolingType<InputType, OutputType>::Backward(
  const InputType& /* input */,
  const OutputType& gy,
  OutputType& g)
{
  arma::Cube<typename OutputType::elem_type> mappedError(
      ((OutputType&) gy).memptr(), outputDimensions[0], outputDimensions[1],
      batchSize * channels, false, true);

  arma::Cube<typename OutputType::elem_type> gTemp(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], channels * batchSize,
      false, true);

  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    Unpooling(inputTemp.slice(s), mappedError.slice(s), gTemp.slice(s));
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MeanPoolingType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(floor));
  ar(CEREAL_NVP(outputDimensions));
  ar(CEREAL_NVP(offset));

  if (Archive::is_loading::value)
    inputTemp.clear();
}

} // namespace ann
} // namespace mlpack

#endif
