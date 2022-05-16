/**
 * @file methods/ann/layer/mean_pooling_impl.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 * @author Shubham Agrawal
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

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType(
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const bool floor) :
    Layer<MatType>(),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    floor(floor),
    channels(0)
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType(
    const MeanPoolingType& other) :
    Layer<MatType>(other),
    kernelWidth(other.kernelWidth),
    kernelHeight(other.kernelHeight),
    strideWidth(other.strideWidth),
    strideHeight(other.strideHeight),
    floor(other.floor),
    channels(other.channels),
    pooling(other.pooling)
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType(
    MeanPoolingType&& other) :
    Layer<MatType>(std::move(other)),
    kernelWidth(std::move(other.kernelWidth)),
    kernelHeight(std::move(other.kernelHeight)),
    strideWidth(std::move(other.strideWidth)),
    strideHeight(std::move(other.strideHeight)),
    floor(std::move(other.floor)),
    channels(std::move(other.channels)),
    pooling(std::move(other.pooling))
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>&
MeanPoolingType<MatType>::operator=(const MeanPoolingType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    kernelWidth = other.kernelWidth;
    kernelHeight = other.kernelHeight;
    strideWidth = other.strideWidth;
    strideHeight = other.strideHeight;
    floor = other.floor;
    channels = other.channels;
    pooling = other.pooling;
  }

  return *this;
}

template<typename MatType>
MeanPoolingType<MatType>&
MeanPoolingType<MatType>::operator=(MeanPoolingType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    kernelWidth = std::move(other.kernelWidth);
    kernelHeight = std::move(other.kernelHeight);
    strideWidth = std::move(other.strideWidth);
    strideHeight = std::move(other.strideHeight);
    floor = std::move(other.floor);
    channels = std::move(other.channels);
    pooling = std::move(other.pooling);
  }

  return *this;
}

template<typename MatType>
void MeanPoolingType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  arma::Cube<typename MatType::elem_type> inputTemp(
      const_cast<MatType&>(input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], input.n_cols * channels, false, false);

  arma::Cube<typename MatType::elem_type> outputTemp(output.memptr(),
      this->outputDimensions[0], this->outputDimensions[1],
      input.n_cols * channels, false, true);

  PoolingOperation(inputTemp, outputTemp);
}

template<typename MatType>
void MeanPoolingType<MatType>::Backward(
  const MatType& input,
  const MatType& gy,
  MatType& g)
{
  arma::Cube<typename MatType::elem_type> mappedError =
      arma::Cube<typename MatType::elem_type>(((MatType&) gy).memptr(),
      this->outputDimensions[0], this->outputDimensions[1],
      channels * input.n_cols, false, false);

  arma::Cube<typename MatType::elem_type> gTemp(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1],
      channels * input.n_cols, false, true);

  gTemp.zeros();
  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    Unpooling(mappedError.slice(s), gTemp.slice(s));
  }
}

template<typename MatType>
void MeanPoolingType<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;

  // Compute the size of the output.
  if (floor)
  {
    this->outputDimensions[0] = std::floor((this->inputDimensions[0] -
        (double) kernelWidth) / (double) strideWidth + 1);
    this->outputDimensions[1] = std::floor((this->inputDimensions[1] -
        (double) kernelHeight) / (double) strideHeight + 1);
  }
  else
  {
    this->outputDimensions[0] = std::ceil((this->inputDimensions[0] -
        (double) kernelWidth) / (double) strideWidth + 1);
    this->outputDimensions[1] = std::ceil((this->inputDimensions[1] -
        (double) kernelHeight) / (double) strideHeight + 1);
  }

  // Higher dimensions are not modified.

  // Cache input size and output size.
  channels = 1;
  for (size_t i = 2; i < this->inputDimensions.size(); ++i)
    channels *= this->inputDimensions[i];
}

template<typename MatType>
template<typename Archive>
void MeanPoolingType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(channels));
  ar(CEREAL_NVP(floor));
}

} // namespace ann
} // namespace mlpack

#endif
