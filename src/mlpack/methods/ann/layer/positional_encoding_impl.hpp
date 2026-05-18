/**
 * @file methods/ann/layer/positional_encoding_impl.hpp
 * @author Kumar Utkarsh
 *
 * Definition of the Positional Encoding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_IMPL_HPP

// In case it hasn't yet been included.
#include "positional_encoding.hpp"

namespace mlpack {

template<typename MatType>
PositionalEncoding<MatType>::PositionalEncoding() :
    Layer<MatType>(),
    embedDim(0),
    maxSequenceLength(0)
{
  // Nothing to do here.
}

template<typename MatType>
PositionalEncoding<MatType>::PositionalEncoding(
    const size_t embedDim,
    const size_t maxSequenceLength) :
    Layer<MatType>(),
    embedDim(embedDim), // This will be set by ComputeOutputDimensions().
    maxSequenceLength(maxSequenceLength)
{
  // Nothing to do.
}

template<typename MatType>
PositionalEncoding<MatType>::PositionalEncoding(
    const PositionalEncoding& layer) :
    Layer<MatType>(layer),
    embedDim(layer.embedDim),
    maxSequenceLength(layer.maxSequenceLength)
{
  // Nothing to do here.
}

template<typename MatType>
PositionalEncoding<MatType>::PositionalEncoding(
    PositionalEncoding&& layer) :
    Layer<MatType>(std::move(layer)),
    embedDim(std::move(layer.embedDim)),
    maxSequenceLength(std::move(layer.maxSequenceLength))
{
  // Reset parameters of other layer.
  layer.embedDim = 0;
  layer.maxSequenceLength = 0;
}

template<typename MatType>
PositionalEncoding<MatType>&
PositionalEncoding<MatType>::operator=(
    const PositionalEncoding& layer)
{
  if (this != &layer)
  {
    Layer<MatType>::operator=(layer);
    embedDim = layer.embedDim;
    maxSequenceLength = layer.maxSequenceLength;
  }

  return *this;
}

template<typename MatType>
PositionalEncoding<MatType>&
PositionalEncoding<MatType>::operator=(
    PositionalEncoding&& layer)
{
  if (this != &layer)
  {
    Layer<MatType>::operator=(std::move(layer));
    embedDim = std::move(layer.embedDim);
    maxSequenceLength = std::move(layer.maxSequenceLength);

    // Reset parameters of other layer.
    layer.embedDim = 0;
    layer.maxSequenceLength = 0;
  }

  return *this;
}

template<typename MatType>
void PositionalEncoding<MatType>::Forward(
    const MatType& input, MatType& output)
{
  if (input.n_rows != embedDim * maxSequenceLength)
    Log::Fatal << "Incorrect input dimensions!" << std::endl;

   positionalEncoding.set_size(maxSequenceLength, embedDim);

  const MatType theta =
    arma::regspace<MatType>(0, maxSequenceLength - 1) *
    arma::exp(
        arma::regspace<MatType>(0, 2, embedDim - 1) *
        (-std::log(10000.0) / embedDim)
    ).t();

  arma::uvec evenCols = arma::regspace<arma::uvec>(0, 2, embedDim - 2);
  arma::uvec oddCols  = arma::regspace<arma::uvec>(1, 2, embedDim - 1);

  positionalEncoding.cols(evenCols) = arma::sin(theta);
  positionalEncoding.cols(oddCols)  = arma::cos(theta);

  output = input.each_col() + arma::vectorise(positionalEncoding.t());
}

template<typename MatType>
void PositionalEncoding<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy;
}

template<typename MatType>
void PositionalEncoding<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;
}

template<typename MatType>
template<typename Archive>
void PositionalEncoding<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(embedDim));
  ar(CEREAL_NVP(maxSequenceLength));
}

} // namespace mlpack

#endif
