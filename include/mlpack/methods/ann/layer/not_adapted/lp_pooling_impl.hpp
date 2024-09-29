/**
 * @file methods/ann/layer/lp_pooling_impl.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Implementation of the lpPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LP_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LP_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "lp_pooling.hpp"

namespace mlpack {

template<typename InputDataType, typename OutputDataType>
LpPooling<InputDataType, OutputDataType>::LpPooling()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
LpPooling<InputDataType, OutputDataType>::LpPooling(
    const size_t normType,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const bool floor) :
    normType(normType),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    floor(floor),
    inSize(0),
    outSize(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(0),
    outputHeight(0),
    reset(false),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void LpPooling<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  batchSize = input.n_cols;
  inSize = input.n_elem / (inputWidth * inputHeight * batchSize);
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&>(input).memptr(),
      inputWidth, inputHeight, batchSize * inSize, false, false);

  if (floor)
  {
    outputWidth = std::floor((inputWidth -
        (double) kernelWidth) / (double) strideWidth + 1);
    outputHeight = std::floor((inputHeight -
        (double) kernelHeight) / (double) strideHeight + 1);
  }
  else
  {
    outputWidth = std::ceil((inputWidth -
        (double) kernelWidth) / (double) strideWidth + 1);
    outputHeight = std::ceil((inputHeight -
        (double) kernelHeight) / (double) strideHeight + 1);
  }

  outputTemp = zeros<arma::Cube<eT> >(outputWidth, outputHeight,
      batchSize * inSize);

  for (size_t s = 0; s < inputTemp.n_slices; s++)
    Pooling(inputTemp.slice(s), outputTemp.slice(s));

  output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem / batchSize,
      batchSize);

  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
  outSize = batchSize * inSize;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void LpPooling<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>& /* input */,
  const arma::Mat<eT>& gy,
  arma::Mat<eT>& g)
{
  arma::cube mappedError = arma::cube(((arma::Mat<eT>&) gy).memptr(),
      outputWidth, outputHeight, outSize, false, false);

  gTemp = zeros<arma::cube>(inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices);

  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    Unpooling(inputTemp.slice(s), mappedError.slice(s), gTemp.slice(s));
  }

  g = arma::mat(gTemp.memptr(), gTemp.n_elem / batchSize, batchSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void LpPooling<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(normType));
  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(floor));
  ar(CEREAL_NVP(inputWidth));
  ar(CEREAL_NVP(inputHeight));
  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(outputHeight));
}

} // namespace mlpack

#endif
