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
  weights.set_size(embeddingSize, vocabSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  output.set_size(embeddingSize * seqLength, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    // ith column of output is a vectorized form of a matrix of shape
    // (embeddingSize, seqLength) selected as a combination of columns from the
    // weights.
    output.col(i) = arma::vectorise(weights.cols(
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
      embeddingSize, seqLength, batchSize, false, false);

  gradient.set_size(arma::size(weights));
  gradient.zeros();

  for (size_t i = 0; i < batchSize; ++i)
  {
    gradient.cols(arma::conv_to<arma::uvec>::from(input.col(i)) - 1)
        += errorTemp.slice(i);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Lookup<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(vocabSize);
  ar & BOOST_SERIALIZATION_NVP(embeddingSize);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    weights.set_size(embeddingSize, vocabSize);
}

} // namespace ann
} // namespace mlpack

#endif
