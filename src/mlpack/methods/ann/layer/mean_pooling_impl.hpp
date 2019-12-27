/**
 * @file mean_pooling_impl.hpp
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

template<typename InputDataType, typename OutputDataType>
MeanPooling<InputDataType, OutputDataType>::MeanPooling()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
MeanPooling<InputDataType, OutputDataType>::MeanPooling(
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const bool floor,
    const size_t padW,
    const size_t padH,
    const std::string paddingType) :
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    floor(floor),
    padWLeft(padW),
    padWRight(padW),
    padHBottom(padH),
    padHTop(padH),
    inSize(0),
    outSize(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(0),
    outputHeight(0),
    reset(false),
    deterministic(false),
    offset(0),
    batchSize(0)
{
  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = ann::Padding<>(padWLeft, padWRight, padHTop, padHBottom);
}

template<typename InputDataType, typename OutputDataType>
MeanPooling<InputDataType, OutputDataType>::MeanPooling(
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const bool floor,
    const std::tuple<size_t, size_t> padW,
    const std::tuple<size_t, size_t> padH,
    const std::string paddingType) :
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    floor(floor),
    padWLeft(std::get<0>(padW)),
    padWRight(std::get<1>(padW)),
    padHBottom(std::get<1>(padH)),
    padHTop(std::get<0>(padH)),
    inSize(0),
    outSize(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(0),
    outputHeight(0),
    reset(false),
    deterministic(false),
    offset(0),
    batchSize(0)
{
  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = ann::Padding<>(padWLeft, padWRight, padHTop, padHBottom);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MeanPooling<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  batchSize = input.n_cols;
  inSize = input.n_elem / (inputWidth * inputHeight * batchSize);
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&>(input).memptr(),
      inputWidth, inputHeight, batchSize * inSize, false, false);

  if (floor)
  {
    outputWidth = std::floor((inputWidth + padWLeft + padWRight -
        (double) kernelWidth) / (double) strideWidth + 1);
    outputHeight = std::floor((inputHeight + padHTop + padHBottom -
        (double) kernelHeight) / (double) strideHeight + 1);

    offset = 0;
  }
  else
  {
    outputWidth = std::ceil((inputWidth + padWLeft + padWRight -
        (double) kernelWidth) / (double) strideWidth + 1);
    outputHeight = std::ceil((inputHeight + padHTop + padHBottom -
        (double) kernelHeight) / (double) strideHeight + 1);

    offset = 1;
  }

  outputTemp = arma::zeros<arma::Cube<eT> >(outputWidth, outputHeight,
      batchSize * inSize);

  bool isPadded {padWLeft != 0 || padWRight != 0 ||
      padHTop != 0 || padHBottom != 0};

  if (isPadded)
  {
    inputPaddedTemp.set_size(inputTemp.n_rows + padWLeft + padWRight,
        inputTemp.n_cols + padHTop + padHBottom, inputTemp.n_slices);

    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      padding.Forward(std::move(inputTemp.slice(i)),
          std::move(inputPaddedTemp.slice(i)));
    }

    for (size_t s = 0; s < inputPaddedTemp.n_slices; s++)
    {
      Pooling(inputPaddedTemp.slice(s), outputTemp.slice(s));
    }
  }
  else
  {
    for (size_t s = 0; s < inputTemp.n_slices; s++)
    {
      Pooling(inputTemp.slice(s), outputTemp.slice(s));
    }
  }

  output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem / batchSize,
      batchSize);

  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
  outSize = batchSize * inSize;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MeanPooling<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>& /* input */,
  const arma::Mat<eT>& gy,
  arma::Mat<eT>& g)
{
  arma::cube mappedError = arma::cube(((arma::Mat<eT>&) gy).memptr(),
      outputWidth, outputHeight, outSize, false, false);

  gTemp = arma::zeros<arma::cube>(inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices);

  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    Unpooling(inputTemp.slice(s), mappedError.slice(s), gTemp.slice(s));
  }

  g = arma::mat(gTemp.memptr(), gTemp.n_elem / batchSize, batchSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanPooling<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(kernelWidth);
  ar & BOOST_SERIALIZATION_NVP(kernelHeight);
  ar & BOOST_SERIALIZATION_NVP(strideWidth);
  ar & BOOST_SERIALIZATION_NVP(strideHeight);
  ar & BOOST_SERIALIZATION_NVP(padWLeft);
  ar & BOOST_SERIALIZATION_NVP(padWRight);
  ar & BOOST_SERIALIZATION_NVP(padHBottom);
  ar & BOOST_SERIALIZATION_NVP(padHTop);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(floor);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);
}

template<typename InputDataType, typename OutputDataType>
void MeanPooling<InputDataType, OutputDataType>::InitializeSamePadding()
{
  /*
   * Using O = (W - F + 2P) / s + 1;
   */
  size_t totalVerticalPadding = (strideWidth - 1) * inputWidth + kernelWidth -
      strideWidth;
  size_t totalHorizontalPadding = (strideHeight - 1) * inputHeight +
      kernelHeight - strideHeight;

  padWLeft = totalVerticalPadding / 2;
  padWRight = totalVerticalPadding - totalVerticalPadding / 2;
  padHTop = totalHorizontalPadding / 2;
  padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;
}

} // namespace ann
} // namespace mlpack

#endif
