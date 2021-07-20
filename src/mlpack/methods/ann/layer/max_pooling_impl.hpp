/**
 * @file methods/ann/layer/max_pooling_impl.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Implementation of the MaxPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MAX_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MAX_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "max_pooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
MaxPoolingType<InputType, OutputType>::MaxPoolingType()
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
MaxPoolingType<InputType, OutputType>::MaxPoolingType(
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
    reset(false),
    offset(0),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void MaxPoolingType<InputType, OutputType>::Forward(
  const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  arma::Cube<typename InputType::elem_type> inputTemp(
      const_cast<InputType&>(input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], batchSize * channels, false, false);

  const std::vector<size_t> outputDimensions = OutputDimensions();
  arma::Cube<typename OutputType::elem_type> outputTemp(output.memptr(),
      outputDimensions[0], outputDimensions[1], batchSize * channels, false,
      true);

  if (this->training)
  {
    poolingIndices.push_back(outputTemp);
  }

  if (!reset)
  {
    const size_t elements = this->inputDimensions[0] * this->inputDimensions[1];
    indicesCol = arma::linspace<arma::Col<size_t> >(0, (elements - 1),
        elements);

    indices = arma::Mat<size_t>(indicesCol.memptr(), this->inputDimensions[0],
        this->inputDimensions[1]);

    reset = true;
  }

  for (size_t s = 0; s < inputTemp.n_slices; s++)
  {
    if (this->training)
    {
      PoolingOperation(inputTemp.slice(s), outputTemp.slice(s),
          poolingIndices.back().slice(s));
    }
    else
    {
      PoolingOperation(inputTemp.slice(s), outputTemp.slice(s),
          inputTemp.slice(s));
    }
  }
}

template<typename InputType, typename OutputType>
void MaxPoolingType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  const std::vector<size_t> outputDimensions = OutputDimensions();
  arma::Cube<typename OutputType::elem_type> mappedError =
      arma::Cube<typename OutputType::elem_type>(((OutputType&) gy).memptr(),
      outputDimensions[0], outputDimensions[1], channels * batchSize, false,
      false);

  arma::Cube<typename OutputType::elem_type> gTemp(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], channels * batchSize,
      false, true);

  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    Unpooling(mappedError.slice(s), gTemp.slice(s),
        poolingIndices.back().slice(s));
  }

  poolingIndices.pop_back();
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MaxPoolingType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(channels));
  ar(CEREAL_NVP(outputDimensions));
  ar(CEREAL_NVP(floor));
  ar(CEREAL_NVP(offset));

  if (Archive::is_loading::value)
    reset = false;
}

} // namespace ann
} // namespace mlpack

#endif
