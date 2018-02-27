/**
 * @file max_pooling_impl.hpp
 * @author Manthan-R-Sheth
 *
 * Implementation of the Embedding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_EMBEDDING_IMPL_HPP
#define MLPACK_EMBEDDING_IMPL_HPP

// In case it hasn't yet been included.
#include "embedding.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Embedding<InputDataType, OutputDataType>::Embedding()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
Embedding<InputDataType, OutputDataType>::Embedding(
  const size_t vocabSize,
  const size_t dimensionSize,
  const bool pretrain) :
  vocabSize(vocabSize),
  dimensionSize(dimensionSize),
  pretrain(pretrain)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
void Embedding<InputDataType, OutputDataType>::Reset()
{
  embeddingMatrix = arma::mat(vocabSize, dimensionSize,arma::fill::randu);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Embedding<InputDataType, OutputDataType>::Forward(
  const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{

  // Apply embeddingMatrix multiplication to the input and store the results.
  for (arma::uword j = 0; j < input.n_cols; ++j)
  {
    arma::rowvec wordRepresentation = embeddingMatrix.row(input(0,j));
    output(arma::span(j,j),arma::span::all) = wordRepresentation;
  }

}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Embedding<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if (!pretrain)
  {
    g = embeddingMatrix.t() * gy;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Embedding<InputDataType, OutputDataType>::Gradient(
  const arma::Mat<eT>&& input,
  arma::Mat<eT>&& error,
  arma::Mat<eT>&& gradient)
{
  if (!pretrain)
    gradient.submat(0, 0, embeddingMatrix.n_elem - 1, 0) = arma::vectorise(
      error * input.t());
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Embedding<InputDataType, OutputDataType>::serialize(
  Archive& ar,
  const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(vocabSize);
  ar & BOOST_SERIALIZATION_NVP(dimensionSize);
  ar & BOOST_SERIALIZATION_NVP(pretrain);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    embeddingMatrix.set_size(vocabSize , dimensionSize);
}

} // namespace ann
} // namespace mlpack

#endif //MLPACK_EMBEDDING_IMPL_HPP
