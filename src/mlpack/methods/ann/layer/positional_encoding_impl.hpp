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

template<typename MatType, typename RegularizerType>
PositionalEncoding<MatType, RegularizerType>::PositionalEncoding() :
    Layer<MatType>(),
    embedDim(0),
    maxSequenceLength(0)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
PositionalEncoding<MatType, RegularizerType>::PositionalEncoding(
    const size_t embedDim,
    const size_t maxSequenceLength,
    RegularizerType regularizer) :
    Layer<MatType>(),
    embedDim(embedDim), // This will be set by ComputeOutputDimensions().
    maxSequenceLength(maxSequenceLength),
    regularizer(regularizer)
{
  // Nothing to do.
}

template<typename MatType, typename RegularizerType>
PositionalEncoding<MatType, RegularizerType>::PositionalEncoding(
    const PositionalEncoding& layer) :
    Layer<MatType>(layer),
    embedDim(layer.embedDim),
    maxSequenceLength(layer.maxSequenceLength),
    regularizer(layer.regularizer)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
PositionalEncoding<MatType, RegularizerType>::PositionalEncoding(
    PositionalEncoding&& layer) :
    Layer<MatType>(std::move(layer)),
    embedDim(std::move(layer.embedDim)),
    maxSequenceLength(std::move(layer.maxSequenceLength)),
    regularizer(std::move(layer.regularizer))
{
  // Reset parameters of other layer.
  layer.embedDim = 0;
  layer.maxSequenceLength = 0;
}

template<typename MatType, typename RegularizerType>
PositionalEncoding<MatType, RegularizerType>&
PositionalEncoding<MatType, RegularizerType>::operator=(
    const PositionalEncoding& layer)
{
  if (this != &layer)
  {
    Layer<MatType>::operator=(layer);
    embedDim = layer.embedDim;
    maxSequenceLength = layer.maxSequenceLength;
    regularizer = layer.regularizer;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
PositionalEncoding<MatType, RegularizerType>&
PositionalEncoding<MatType, RegularizerType>::operator=(
    PositionalEncoding&& layer)
{
  if (this != &layer)
  {
    Layer<MatType>::operator=(std::move(layer));
    embedDim = std::move(layer.embedDim);
    maxSequenceLength = std::move(layer.maxSequenceLength);
    regularizer = std::move(layer.regularizer);

    // Reset parameters of other layer.
    layer.embedDim = 0;
    layer.maxSequenceLength = 0;
  }

  return *this;
}

template<typename InputType, typename OutputType>
void PositionalEncoding<InputType, OutputType>::InitPositionalEncoding()
{
  positionalEncoding.set_size(maxSequenceLength, embedDim);
  const InputType position = arma::regspace(0, 1, maxSequenceLength - 1);
  const InputType divTerm = exp(arma::regspace(0, 2, embedDim - 1)
      * (- std::log(10000.0) / embedDim));
  const InputType theta = position * divTerm.t();
  for (size_t i = 0; i < theta.n_cols; ++i)
  {
    positionalEncoding.col(2 * i) = arma::sin(theta.col(i));
    positionalEncoding.col(2 * i + 1) = arma::cos(theta.col(i));
  }
  positionalEncoding = vectorise(positionalEncoding.t());
}

template<typename MatType, typename RegularizerType>
void PositionalEncoding<MatType, RegularizerType>::SetWeights(
    const MatType& weights)
{
  MakeAlias(positionalEncoding, weights, maxSequenceLength, embedDim);
}

template<typename MatType, typename RegularizerType>
void PositionalEncoding<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
{
  if (input.n_rows != embedDim * maxSequenceLength)
    Log::Fatal << "Incorrect input dimensions!" << std::endl;

  output = input.each_col() + positionalEncoding;
}

template<typename MatType, typename RegularizerType>
void PositionalEncoding<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy;
}

template<typename MatType, typename RegularizerType>
void PositionalEncoding<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
void PositionalEncoding<MatType, RegularizerType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;
}

template<typename MatType, typename RegularizerType>
template<typename Archive>
void PositionalEncoding<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(embedDim));
  ar(CEREAL_NVP(maxSequenceLength));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
