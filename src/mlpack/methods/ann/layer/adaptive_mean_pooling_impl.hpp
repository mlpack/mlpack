/**
 * @file adaptive_mean_pooling_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of the Adaptive Mean Pooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MEAN_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MEAN_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "adaptive_mean_pooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
AdaptiveMeanPooling<InputDataType, OutputDataType>::AdaptiveMeanPooling()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
AdaptiveMeanPooling<InputDataType, OutputDataType>::AdaptiveMeanPooling(
    const size_t outputWidth,
    const size_t outputHeight) :
    inSize(0),
    outSize(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(outputWidth),
    outputHeight(outputHeight),
    reset(false),
    deterministic(false),
    batchSize(0)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
AdaptiveMeanPooling<InputDataType, OutputDataType>::AdaptiveMeanPooling(
    const std::tuple<size_t, size_t> outputShape):
    inSize(0),
    outSize(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(std::get<0>(outputShape)),
    outputHeight(std::get<1>(outputShape)),
    reset(false),
    deterministic(false),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void AdaptiveMeanPooling<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  IntializeAdaptivePadding();
  batchSize = input.n_cols;
  inSize = input.n_elem / (inputWidth * inputHeight);
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&&>(input).memptr(),
      inputWidth, inputHeight, batchSize * inSize, false, false);
  outputTemp = arma::zeros<arma::Cube<eT> >(outputWidth, outputHeight,
      batchSize * inSize);
  for (size_t s = 0; s < inputTemp.n_slices; s++)
    Pooling(inputTemp.slice(s), outputTemp.slice(s));

  output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem / batchSize,
      batchSize);

  outSize = batchSize * inSize;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void AdaptiveMeanPooling<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */,
  arma::Mat<eT>&& gy,
  arma::Mat<eT>&& g)
{
  arma::cube mappedError = arma::cube(gy.memptr(), outputWidth,
      outputHeight, outSize, false, false);

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
void AdaptiveMeanPooling<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(kernelWidth);
  ar & BOOST_SERIALIZATION_NVP(kernelHeight);
  ar & BOOST_SERIALIZATION_NVP(strideWidth);
  ar & BOOST_SERIALIZATION_NVP(strideHeight);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);
}

} // namespace ann
} // namespace mlpack

#endif