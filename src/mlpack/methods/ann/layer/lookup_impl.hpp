/**
 * @file methods/ann/layer/lookup_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Lookup (embedding) layer class.
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

template<typename MatType, typename RegularizerType>
Lookup<MatType, RegularizerType>::Lookup() :
    Layer<MatType>(),
    vocabSize(0),
    embeddingSize(0)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
Lookup<MatType, RegularizerType>::Lookup(
    const size_t vocabSize, 
    const size_t embeddingSize,
    RegularizerType regularizer) :
    Layer<MatType>(),
    vocabSize(0),
    embeddingSize(0),
    regularizer(regularizer)
{
  // Nothing to do here.
}

// Copy constructor.
template<typename MatType, typename RegularizerType>
Lookup<MatType, RegularizerType>::Lookup(const Lookup& layer) :
    Layer<MatType>(layer),
    vocabSize(layer.vocabSize),
    embeddingSize(layer.embeddingSize),
    regularizer(layer.regularizer)
{
  // Nothing else to do.
}

// Move constructor.
template<typename MatType, typename RegularizerType>
Lookup<MatType, RegularizerType>::Lookup(Lookup&& layer) :
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
Lookup<MatType, RegularizerType>&
Lookup<MatType, RegularizerType>::operator=(const Lookup& layer)
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
Lookup<MatType, RegularizerType>&
Lookup<MatType, RegularizerType>::operator=(
    Lookup&& layer)
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
void Lookup<MatType, RegularizerType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, vocabSize, embeddingSize);
}

template<typename MatType, typename RegularizerType>
void Lookup<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
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

template<typename MatType, typename RegularizerType>
void Lookup<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  Log::Fatal << "Lookup cannot be used as an intermediate layer." << std::endl;
}

template<typename MatType, typename RegularizerType>
void Lookup<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  const size_t seqLength = input.n_rows;
  const size_t batchSize = input.n_cols;

  const CubeType errorTemp;
  MakeAlias(const_cast<CubeType&>(errorTemp), error, seqLength, embeddingSize,
      batchSize, 0, false);

  gradient.set_size(arma::size(weights));
  gradient.zeros();

  for (size_t i = 0; i < batchSize; ++i)
  {
    gradient.rows(arma::conv_to<arma::uvec>::from(input.col(i)) - 1)
        += errorTemp.slice(i);
  }
}

template<typename MatType, typename RegularizerType>
void Lookup<MatType, RegularizerType>::ComputeOutputDimensions()
{
  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
    1);

  // The Linear layer flattens its input.
  this->outputDimensions[0] = this->embeddingSize * this->inputDimensions[0];

}

template<typename MatType, typename RegularizerType>
template<typename Archive>
void Lookup<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(vocabSize));
  ar(CEREAL_NVP(embeddingSize));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
