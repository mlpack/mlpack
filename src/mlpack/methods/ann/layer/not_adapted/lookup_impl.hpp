/**
 * @file methods/ann/layer/lookup_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Lookup class a particular convolution, where the width
 * of the convolution is 1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP

// In case it hasn't yet been included.
#include "lookup.hpp"

namespace mlpack {

template <typename InputType, typename OutputDataType>
LookupType<InputType, OutputDataType>::LookupType(
    const size_t vocabSize,
    const size_t embeddingSize) :
    vocabSize(vocabSize),
    embeddingSize(embeddingSize)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
void LookupType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  weights = OutputType(weightsPtr, embeddingSize, vocabSize, false, true);
}

template<typename InputType, typename OutputType>
void LookupType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  for (size_t i = 0; i < batchSize; ++i)
  {
    // ith column of output is a vectorized form of a matrix of shape
    // (embeddingSize, seqLength) selected as a combination of columns from the
    // weights.
    output.col(i) = vectorise(weights.cols(
        ConvTo<arma::uvec>::From(input.col(i)) - 1));
  }
}

template<typename InputType, typename OutputType>
void LookupType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& /* gy */,
    OutputType& /* g */)
{
  Log::Fatal << "Lookup cannot be used as an intermediate layer." << std::endl;
}

template<typename InputType, typename OutputType>
void LookupType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  using CubeType = arma::Cube<typename OutputType::elem_type>;
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  const CubeType errorTemp(const_cast<OutputType&>(error).memptr(),
      embeddingSize, seqLength, batchSize, false, false);

  gradient.zeros();
  for (size_t i = 0; i < batchSize; ++i)
  {
    gradient.cols(ConvTo<arma::uvec>::From(input.col(i)) - 1)
        += errorTemp.slice(i);
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void LookupType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(vocabSize));
  ar(CEREAL_NVP(embeddingSize));
}

} // namespace mlpack

#endif
