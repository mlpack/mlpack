/**
 * @file embedding_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Embedding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_EMBEDDING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_EMBEDDING_IMPL_HPP

// In case it hasn't yet been included.
#include "embedding.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
Embedding<InputDataType, OutputDataType, InitializerType>
::Embedding()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
Embedding<InputDataType, OutputDataType, InitializerType>
::Embedding(const size_t dictionarySize,
    const size_t embeddingDim,
    const int paddingIndex,
    const bool freeze) :
    dictionarySize(dictionarySize),
    embeddingDim(embeddingDim),
    freeze(freeze)
{
  typedef typename InputDataType::elem_type ElemType;
  if (paddingIndex)
  {
    if (paddingIndex > 0)
    {
      Log::Assert(paddingIndex < this->embeddingDim,
          "paddingIndex must be less than embeddingDim");
      this->paddingIndex = paddingIndex;
    }
    else
    {
      Log::Assert(paddingIndex >= - this->embeddingDim,
          "paddingIndex must be less than embeddingDim");
      this->paddingIndex = paddingIndex + this->embeddingDim;
    }
  }
  else
    this->paddingIndex = paddingIndex;
  this->weights.set_size(dictionarySize, embeddingDim);
  ResetParameters();
}

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
void Embedding<InputDataType, OutputDataType, InitializerType>
::ResetParameters()
{
  typedef typename InputDataType::elem_type ElemType;
  InitializerType init;
  init.Initialize(weights, weights.n_rows, weights.n_cols);
  if (paddingIndex)
  {
    weights[paddingIndex] = arma::zeros<arma::Row<ElemType>>(weights.n_cols);
  }
}

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
template<typename InputType, typename OutputType>
void Embedding<InputDataType, OutputDataType, InitializerType>
::Forward(const InputType& input, OutputType& output)
{
  output.set_size(input.n_cols * embeddingDim, input.n_rows);
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    output.col(i) = arma::vectorise(weights.elem(input[i]));
  }
}

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
template<typename eT>
void Embedding<InputDataType, OutputDataType, InitializerType>
::Backward(const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = gy;
}

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
template<typename eT>
void Embedding<InputDataType, OutputDataType, InitializerType>
::Gradient(const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  gradient = arma::zeros<arma::Mat<eT>>(weights.n_rows, weights.n_cols);
  if (!freeze)
    gradient.cols(input) = error;
}

template<typename InputDataType, typename OutputDataType,
         typename InitializerType>
template<typename Archive>
void Embedding<InputDataType, OutputDataType, InitializerType>
::serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(dictionarySize);
  ar & BOOST_SERIALIZATION_NVP(embeddingDim);
  ar & BOOST_SERIALIZATION_NVP(paddingIndex);
  ar & BOOST_SERIALIZATION_NVP(freeze);
}

} // namespace ann
} // namespace mlpack

#endif
