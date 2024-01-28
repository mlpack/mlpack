/**
 * @file methods/ann/layer/positional_encoding_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Positional Encoding.
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

template<typename InputType, typename OutputType>
PositionalEncodingType<InputType, OutputType>::PositionalEncodingType() :
    embedDim(0),
    maxSequenceLength(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
PositionalEncodingType<InputType, OutputType>::PositionalEncodingType(
    const size_t embedDim,
    const size_t maxSequenceLength) :
    embedDim(embedDim),
    maxSequenceLength(maxSequenceLength)
{
  InitPositionalEncoding();
}

template<typename InputType, typename OutputType>
void PositionalEncodingType<InputType, OutputType>::InitPositionalEncoding()
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

template<typename InputType, typename OutputType>
void PositionalEncodingType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (input.n_rows != embedDim * maxSequenceLength)
    Log::Fatal << "Incorrect input dimensions!" << std::endl;

  output = input.each_col() + positionalEncoding;
}

template<typename InputType, typename OutputType>
void PositionalEncodingType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = gy;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void PositionalEncodingType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(embedDim));
  ar(CEREAL_NVP(maxSequenceLength));

  if (cereal::is_loading<Archive>())
    InitPositionalEncoding();
}

} // namespace mlpack

#endif
