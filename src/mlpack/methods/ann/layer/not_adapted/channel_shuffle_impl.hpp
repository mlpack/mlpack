/**
 * @file methods/ann/layer/channe_shuffle_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the channel shuffle function as an individual layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CHANNEL_SHUFFLE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CHANNEL_SHUFFLE_IMPL_HPP

// In case it hasn't yet been included.
#include "channel_shuffle.hpp"

namespace mlpack {


template<typename InputDataType, typename OutputDataType>
ChannelShuffle<InputDataType, OutputDataType>::
ChannelShuffle():
    inRowSize(0),
    inColSize(0),
    depth(0),
    groupCount(0),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
ChannelShuffle<InputDataType, OutputDataType>::
ChannelShuffle(
    const size_t inRowSize,
    const size_t inColSize,
    const size_t depth,
    const size_t groupCount):
    inRowSize(inRowSize),
    inColSize(inColSize),
    depth(depth),
    groupCount(groupCount),
    batchSize(0)
{
  if (depth % groupCount != 0)
  {
    Log::Fatal << "Number of channels must be divisible by groupCount!"
        << std::endl;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ChannelShuffle<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  batchSize = input.n_cols;

  if (output.is_empty())
    output.set_size(inRowSize * inColSize * depth, batchSize);
  else
  {
    assert(output.n_rows == inRowSize * inColSize * depth);
    assert(output.n_cols == batchSize);
  }

  arma::cube inputAsCube(const_cast<arma::Mat<eT>&>(input).memptr(),
      inRowSize, inColSize, depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), inRowSize, inColSize,
      depth * batchSize, false, true);

  const size_t groupSize = depth / groupCount;
  size_t outChannelIdx = 0;
  for (size_t k = 0; k < batchSize; ++k)
  {
    for (size_t i = 0; i < groupSize; ++i)
    {
      for (size_t g = 0; g < groupCount; ++g, ++outChannelIdx)
      {
        size_t inChannelIdx = k * batchSize + g * groupSize + i;
        outputAsCube.slice(outChannelIdx) = inputAsCube.slice(inChannelIdx);
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ChannelShuffle<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /*input*/,
    const arma::Mat<eT>& gradient,
    arma::Mat<eT>& output)
{
  if (output.is_empty())
    output.set_size(inRowSize * inColSize * depth, batchSize);
  else
  {
    assert(output.n_rows == inRowSize * inColSize * depth);
    assert(output.n_cols == batchSize);
  }

  arma::cube gradientAsCube(((arma::Mat<eT>&) gradient).memptr(), inColSize,
      inColSize, depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), inRowSize, inColSize,
      depth * batchSize, false, true);

  const size_t groupSize = depth / groupCount;
  size_t gradientChannelIdx = 0;
  for (size_t k = 0; k < batchSize; ++k)
  {
    for (size_t i = 0; i < groupSize; ++i)
    {
      for (size_t g = 0; g < groupCount; ++g, ++gradientChannelIdx)
      {
        size_t outChannelIdx = k * batchSize + g * groupSize + i;
        outputAsCube.slice(outChannelIdx) =
            gradientAsCube.slice(gradientChannelIdx);
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ChannelShuffle<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(inRowSize));
  ar(CEREAL_NVP(inColSize));
  ar(CEREAL_NVP(depth));
}

} // namespace mlpack

#endif
