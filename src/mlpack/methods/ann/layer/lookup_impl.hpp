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
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
Lookup<InputDataType, OutputDataType>::Lookup(
    const size_t vocabSize,
    const size_t embeddingSize) :
    vocabSize(vocabSize),
    embeddingSize(embeddingSize)
{
  weights.set_size(vocabSize, embeddingSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  output.set_size(seqLength * embeddingSize, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    //! ith column of output is a vectorized form of a matrix of shape
    //! (seqLength, embeddingSize) selected as a combination of rows from the
    //! weights. The MultiheadAttention class requires this particular ordering
    //! of matrix dimensions.
    output.col(i) = arma::vectorise(weights.rows(
        arma::conv_to<arma::uvec>::from(input.col(i)) - 1));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& /* gy */,
    arma::Mat<eT>& /* g */)
{
  Log::Fatal << "Lookup cannot be used as an intermediate layer." << std::endl;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  arma::Cube<eT> errorTemp(const_cast<arma::Mat<eT>&>(error).memptr(),
      seqLength, embeddingSize, batchSize, false, false);

  gradient.set_size(arma::size(weights));
  gradient.zeros();

  for (size_t i = 0; i < batchSize; ++i)
  {
    gradient.rows(arma::conv_to<arma::uvec>::from(input.col(i)) - 1)
        += errorTemp.slice(i);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Lookup<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(vocabSize);
  ar & CEREAL_NVP(embeddingSize);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    weights.set_size(vocabSize, embeddingSize);
}

} // namespace ann
} // namespace mlpack

#endif
