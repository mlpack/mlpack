/**
 * @file methods/ann/layer/embedding_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Embedding layer class.
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

template<typename MatType, typename RegularizerType>
Embedding<MatType, RegularizerType>::Embedding() :
    Layer<MatType>(),
    vocabSize(0),
    embeddingSize(0)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
Embedding<MatType, RegularizerType>::Embedding(
    const size_t vocabSize,
    const size_t embeddingSize,
    RegularizerType regularizer) :
    Layer<MatType>(),
    vocabSize(vocabSize),
    embeddingSize(embeddingSize),
    regularizer(regularizer)
{
  // Nothing to do here.
}

// Copy constructor.
template<typename MatType, typename RegularizerType>
Embedding<MatType, RegularizerType>::Embedding(const Embedding& layer) :
    Layer<MatType>(layer),
    vocabSize(layer.vocabSize),
    embeddingSize(layer.embeddingSize),
    regularizer(layer.regularizer)
{
  // Nothing else to do.
}

// Move constructor.
template<typename MatType, typename RegularizerType>
Embedding<MatType, RegularizerType>::Embedding(Embedding&& layer) :
    Layer<MatType>(std::move(layer)),
    vocabSize(std::move(layer.vocabSize)),
    embeddingSize(std::move(layer.embeddingSize)),
    regularizer(std::move(layer.regularizer))
{
  // Reset parameters of other layer.
  layer.vocabSize = 0;
  layer.embeddingSize = 0;
}

template<typename MatType, typename RegularizerType>
Embedding<MatType, RegularizerType>&
Embedding<MatType, RegularizerType>::operator=(const Embedding& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(layer);
    vocabSize = layer.vocabSize;
    embeddingSize = layer.embeddingSize;
    regularizer = layer.regularizer;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
Embedding<MatType, RegularizerType>&
Embedding<MatType, RegularizerType>::operator=(
    Embedding&& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(std::move(layer));
    vocabSize = std::move(layer.vocabSize);
    embeddingSize = std::move(layer.embeddingSize);
    regularizer = std::move(layer.regularizer);

    // Reset parameters of other layer.
    layer.vocabSize = 0;
    layer.embeddingSize = 0;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
void Embedding<MatType, RegularizerType>::SetWeights(const MatType& weightsIn)
{
  // The weights matrix is of size (embeddingSize, vocabSize).  Each vocabulary
  // item's embedding is represented in a single column.
  MakeAlias(weights, weightsIn, embeddingSize, vocabSize);
}

template<typename MatType, typename RegularizerType>
void Embedding<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
{
  const size_t batchSize = input.n_cols;
  for (size_t i = 0; i < batchSize; ++i)
  {
    // ith column of output is a vectorized form of a matrix of shape
    // (seqLength, embeddingSize) selected as a combination of rows from the
    // weights. The MultiheadAttention class requires this particular ordering
    // of matrix dimensions.
    output.col(i) = vectorise(weights.cols(
        arma::conv_to<arma::uvec>::from(input.col(i))));
  }
}

template<typename MatType, typename RegularizerType>
void Embedding<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& /* gy */,
    MatType& g)
{
  // NOTE: Embedding should be the first layer of a network, and shouldn't be
  // used as an intermediate layer!  So Backward() does nothing except set g to
  // zeros.
  g.zeros();
}

template<typename MatType, typename RegularizerType>
void Embedding<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  const CubeType errorTemp;
  MakeAlias(const_cast<CubeType&>(errorTemp), error, embeddingSize, seqLength,
      batchSize, 0, false);
  MatType gradTemp;
  MakeAlias(gradTemp, gradient, embeddingSize, vocabSize);

  gradient.zeros();
  for (size_t j = 0; j < batchSize; ++j)
  {
    for (size_t i = 0; i < seqLength; ++i)
    {
      gradTemp.col((size_t) input(i, j)) += errorTemp.slice(j).col(i);
    }
  }
}

template<typename MatType, typename RegularizerType>
void Embedding<MatType, RegularizerType>::ComputeOutputDimensions()
{
  // The Embedding layer returns a two-dimensional output and flattens its
  // input.
  this->outputDimensions.clear();
  this->outputDimensions.push_back(this->embeddingSize);
  this->outputDimensions.push_back(this->inputDimensions[0]);
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
    this->outputDimensions[1] *= this->inputDimensions[i];
}

template<typename MatType, typename RegularizerType>
template<typename Archive>
void Embedding<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(vocabSize));
  ar(CEREAL_NVP(embeddingSize));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
